package moa.classifiers;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Measurement;
import moa.core.StringUtils;

/**
 * Logistic regression for classification
 */
public class LogisticRegressionClassifier extends AbstractClassifier {

    /**
     * Maximum number of weights to be printed by getModelDescription function
     */
    private static final int MAX_WEIGHTS_TO_PRINT = 10;

    /**
     * Used for serialization
     */
    private static final long serialVersionUID = 1L;

    /**
     * Learning rate used to update the feature weights (not used for the bias).
     */
    public FloatOption weightsLearningRateOption = new FloatOption(
            "weightsLearningRate",
            'r',
            "Learning rate for feature weights.",
            0.01,
            0.0,
            Double.MAX_VALUE
    );

    /**
     * Learning rate used to update the intercept (bias term).
     */
    public FloatOption biasLearningRateOption = new FloatOption(
            "biasLearningRate",
            'b',
            "Learning rate for bias (intercept). If 0, the intercept is not updated.",
            0.01,
            0.0,
            Double.MAX_VALUE
    );

    /**
     * L1 penalization hyper-parameter (Lasso).
     */
    public FloatOption l1Option = new FloatOption(
            "l1Penalty",
            'l', // Lasso
            "L1 regularization factor.",
            0.0,
            0.0,
            Double.MAX_VALUE
    );

    /**
     * L2 penalization hyper-parameter (Ridge).
     */
    public FloatOption l2Option = new FloatOption(
            "l2Penalty",
            'q', // Quadratic penalization
            "L2 regularization factor.",
            0.0,
            0.0,
            Double.MAX_VALUE
    );

    /**
     * Maximum absolute value used to clip the loss gradient.
     */
    public FloatOption clipGradientOption = new FloatOption(
            "clipGradient",
            'c',
            "Maximum absolute value for gradient.",
            1e12,
            0.0,
            Double.MAX_VALUE
    );

    /**
     * Initial value for the bias (intercept).
     */
    public FloatOption initialBiasOption = new FloatOption(
            "initialBias",
            'i', // Intercept
            "Initial value for the bias (intercept).",
            0.0,
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY
    );

    /**
     * Weights of the model.
     */
    protected double[] weights;

    /**
     * Intercept value.
     */
    protected double bias;

    // Part for cumulative L1 penalty state (River implementation)
    /**
     * Global cumulative L1 penalty accumulated across updates.
     */
    protected double maxCumL1;

    /**
     * cumL1[i] stores the cumulative correction already applied to weight i.
     */
    protected double[] cumL1;

    @Override
    public double[] getVotesForInstance(Instance inst) {

        checkInstance(inst);

        //Initialization of the weights + bias (if needed)
        initModelIfNeeded(inst);

        // Compute the score
        double score = rawScore(inst);

        //Compute the sigmoid
        double p1 = sigmoid(score); // probability of the positive class
        double p0 = 1.0 - p1; // probability of the negative class (c0)

        double[] votes = new double[2]; // Binary classifier admits 2 classes
        votes[0] = p0; // class 0
        votes[1] = p1; // class 1
        return votes;
    }

    /**
     * Checks that the instance is binary and, if the model is already initialized, that it has a consistent number of input attributes.
     *
     * @throws IllegalArgumentException if the number of classes is not 2 or if it has inconsistent number of features.
     */
    protected void checkInstance(Instance inst) {
        if (inst.numClasses() != 2) {
            throw new IllegalArgumentException(
                    "LogisticRegressionClassifier supports only binary classification."
            );
        }
        if (this.weights != null && this.weights.length != inst.numInputAttributes()) {
            throw new IllegalArgumentException(
                    "Inconsistent number of input attributes."
            );
        }
    }

    /**
     * Lazily initializes weights and bias based on the number of input attributes.
     * If L1 regularization is enabled, also initializes its cumulative penalty state.
     *
     * @param inst the reference instance.
     */
    protected void initModelIfNeeded(Instance inst) {
        if (this.weights == null) {
            // We will allocate a weight for each feature of the sample (excluding the label)
            this.weights = new double[inst.numInputAttributes()];
            this.bias = this.initialBiasOption.getValue();

            // Initialization of L1 variables
            if (this.l1Option.getValue() != 0.0) {
                this.cumL1 = new double[inst.numInputAttributes()];
                this.maxCumL1 = 0.0;
            }
        }
    }

    /**
     * Computes the linear score of the given instance.
     *
     * @param inst the input instance.
     * @return the linear score z = w · x + b.
     */
    protected double rawScore(Instance inst) {
        double score = this.bias;
        for (int i = 0; i < inst.numInputAttributes(); i++) {
            score += this.weights[i] * inst.valueInputAttribute(i); // dot_product(w,x) + b
        }
        return score;
    }

    /**
     * Computes a numerically stable sigmoid function (to prevent overflow).
     *
     * @param z the input value (linear score).
     * @return the sigmoid of {@code z}.
     */
    protected double sigmoid(double z) {
        if (z >= 0.0) {
            // Standard sigmoid form
            double expNeg = Math.exp(-z);
            return 1.0 / (1.0 + expNeg);
        } else {
            // Alternative sigmoid formula (equivalent to the standard one)
            double expPos = Math.exp(z);
            return expPos / (1.0 + expPos);
        }
    }

    @Override
    public void resetLearningImpl() {
        this.weights = null;
        this.bias = this.initialBiasOption.getValue();

        this.cumL1 = null;
        this.maxCumL1 = 0.0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // Skip training if the class label is missing.
        if (inst.classIsMissing()) {
            return;
        }

        checkInstance(inst);
        checkRegularizationConf();
        initModelIfNeeded(inst);

        double lrWeights = this.weightsLearningRateOption.getValue();
        double lrBias = this.biasLearningRateOption.getValue();

        double l1 = this.l1Option.getValue();
        double l2 = this.l2Option.getValue();

        // Initialize L1 state in case L1 is enabled after model initialization.
        if (l1 != 0.0 && this.cumL1 == null) {
            this.cumL1 = new double[inst.numInputAttributes()];
            this.maxCumL1 = 0.0;
        }

        double y = inst.classValue();
        double instanceWeight = inst.weight(); // Instance weight for weighted updates.

        double score = rawScore(inst);
        double p = sigmoid(score);

        // Loss gradient term used to update both weights and bias
        double logitGradient = (p - y) * instanceWeight; // Gradient of the log-loss with respect to the logit z = W · X + b
        logitGradient = clipGradient(logitGradient);

        // Gradient descent update of the bias (no regularization)
        this.bias -= lrBias * logitGradient;

        // Gradient descent update of the weights
        for (int i = 0; i < inst.numInputAttributes(); i++) {
            double x_i = inst.valueInputAttribute(i);
            double weightGradient = logitGradient * x_i; // Gradient of the log-loss w.r.t. weight w_i

            // w_i <- w_i - lr * (gradient_logLoss + l2 * w_i)
            // We perform gradient descent (we minimize the log-loss)
            this.weights[i] -= lrWeights * (weightGradient + l2 * this.weights[i]);
        }

        // L1 applied to weights
        if (l1 != 0.0) {
            this.maxCumL1 += l1 * lrWeights;
            applyL1(inst);
        }
    }

    /**
     * Applies River-style cumulative L1 penalty to the weights corresponding to the non-zero features of the current instance.
     * This must be applied after the standard weights update. Look at river/linear_model/base.py (class GLM)
     *
     * @param inst the instance used to train the model.
     */
    protected void applyL1(Instance inst) {

        for (int i = 0; i < inst.numInputAttributes(); i++) {
            double x_i = inst.valueInputAttribute(i);

            // Skip zero-valued features.
            if (x_i == 0.0) {
                continue;
            }

            double curr_w_i = this.weights[i]; // Value of the weight before L1 regularization

            if (curr_w_i > 0.0) {
                this.weights[i] = Math.max(0.0, curr_w_i - (this.maxCumL1 + this.cumL1[i]));
            } else if (curr_w_i < 0.0) {
                this.weights[i] = Math.min(0.0, curr_w_i + (this.maxCumL1 - this.cumL1[i]));
            }

            // Update cumulative correction for weight i
            this.cumL1[i] += this.weights[i] - curr_w_i;
        }
    }

    /**
     * Clips a gradient to the range [-clipGradient, clipGradient].
     *
     * @param grad the gradient value.
     * @return the clipped value.
     */
    private double clipGradient(double grad) {
        double clipMaxValue = this.clipGradientOption.getValue();
        return Math.max(-clipMaxValue, Math.min(grad, clipMaxValue));
    }

    /**
     * Checks that at most one regularization penalty is active.
     *
     * @throws UnsupportedOperationException if both L1 and L2 are enabled.
     */
    protected void checkRegularizationConf() {
        if (this.l1Option.getValue() != 0.0 && this.l2Option.getValue() != 0.0) {
            throw new UnsupportedOperationException(
                    "The joint use of L1 and L2 penalties is not supported."
            );
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{
                new Measurement("weightsLearningRate", this.weightsLearningRateOption.getValue()),
                new Measurement("biasLearningRate", this.biasLearningRateOption.getValue()),
                new Measurement("l1Penalty", this.l1Option.getValue()),
                new Measurement("l2Penalty", this.l2Option.getValue()),
                new Measurement("initialBias", this.initialBiasOption.getValue()),
                new Measurement("bias", this.bias),
                new Measurement("numWeights", getNumWeights()),
                new Measurement("nonZeroWeights", getNumNonZeroWeights())
        };
    }

    /**
     * Returns the number of weights, or 0 if the model is not initialized.
     *
     * @return the number of weights.
     */
    private int getNumWeights() {
        return this.weights == null ? 0 : this.weights.length;
    }

    /**
     * Returns the number of non-zero weights in the model.
     *
     * <p>If the model has not been initialized yet, returns 0.
     *
     * @return the number of weights whose value is not zero.
     */
    private int getNumNonZeroWeights() {
        int numNonZeroWeights = 0;

        if (this.weights != null) {
            for (double w : this.weights) {
                if (w != 0.0) {
                    numNonZeroWeights++;
                }
            }
        }

        return numNonZeroWeights;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

        StringUtils.appendIndented(out, indent, "Logistic Regression Classifier\n");

        StringUtils.appendIndented(out, indent + 1,
                "weightsLearningRate: " + this.weightsLearningRateOption.getValue() + "\n");

        StringUtils.appendIndented(out, indent + 1,
                "biasLearningRate: " + biasLearningRateOption.getValue() + "\n");

        StringUtils.appendIndented(out, indent + 1,
                "l1Penalty: " + this.l1Option.getValue() + "\n");

        StringUtils.appendIndented(out, indent + 1,
                "l2Penalty: " + this.l2Option.getValue() + "\n");

        StringUtils.appendIndented(out, indent + 1,
                "initialBias: " + this.initialBiasOption.getValue() + "\n");

        StringUtils.appendIndented(out, indent + 1,
                "bias: " + this.bias + "\n");

        if (this.weights == null) {
            StringUtils.appendIndented(out, indent + 1,
                    "Model not initialized yet.\n");
            return;
        }

        StringUtils.appendIndented(out, indent + 1,
                "numWeights: " + getNumWeights() + "\n");

        StringUtils.appendIndented(out, indent + 1,
                "nonZeroWeights: " + getNumNonZeroWeights() + "\n");

        int maxToPrint = Math.min(MAX_WEIGHTS_TO_PRINT, getNumWeights());

        StringUtils.appendIndented(out, indent + 1,
                "weights (first " + maxToPrint + "):\n");

        for (int i = 0; i < maxToPrint; i++) {
            StringUtils.appendIndented(out, indent + 2,
                    "w[" + i + "] = " + this.weights[i] + "\n");
        }
    }

    @Override
    public boolean isRandomizable() {
        return false; // Logistic regression is deterministic (no randomness)
    }
}