/*
 *    SoftmaxRegression.java
 *    Copyright (C) 2026 Politecnico di Milano, Italy
 *    @author Christian Carstens (christianthomas.carstens@mail.polimi.it)
 *    @author Matteo Gatti (matteo7.gatti@mail.polimi.it)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

package moa.classifiers;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.core.StringUtils;

import com.github.javacliparser.FloatOption;

import com.yahoo.labs.samoa.instances.Instance;

/**
 * Softmax Regression Classifier.
 *
 * <p>Incremental on-line Softmax Regression (Multinomial Logistic Regression) for multi-class classification.</p>
 *
 * <p>
 * This classifier generalizes binary Logistic Regression to multiple classes.
 * It uses Stochastic Gradient Descent (SGD) to optimize the multinomial log loss
 * (cross-entropy) and applies the softmax activation function.
 * </p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-r : Learning rate for feature weights</li>
 * <li>-b : Learning rate for bias intercept</li>
 * <li>-l : L2 regularization penalty</li>
 * <li>-i : Initial bias value</li>
 * </ul>
 *
 * <h3>Algorithm Overview</h3>
 * <ul>
 * <li><b>Forward pass:</b> z_k = w_k · x + b_k; p_k = exp(z_k) / Σ exp(z_j)</li>
 * <li><b>Loss:</b> Cross-entropy = - Σ I(y=k) * log(p_k)</li>
 * <li><b>Gradient:</b> ∇L_k = (p_k - I(y=k))·x (for weights); (p_k - I(y=k)) (for bias)</li>
 * <li><b>Update:</b> w_k ← w_k · (1 - lr·λ) - lr·∇L_k (with L2 weight decay); b_k ← b_k - lr_b·∇L_k (no regularization)</li>
 * </ul>
 *
 * @author Christian Carstens (christianthomas.carstens@mail.polimi.it)
 * @author Matteo Gatti (matteo7.gatti@mail.polimi.it)
 * @version $Revision: 1 $
 */
public class SoftmaxRegression extends AbstractClassifier implements MultiClassClassifier {

    /**
     * Used for serialization
     */
    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Softmax Regression for online multi-class classification using SGD with cross-entropy loss.";
    }

    // ---------------------------------------------------------------
    // MOA Options
    // ---------------------------------------------------------------

    /**
     * Learning rate used to update the feature weights (not used for the bias).
     */
    public FloatOption learningRateOption = new FloatOption(
            "learningRate", 'r',
            "Learning rate for SGD weight updates.",
            0.01, 0.0, Double.MAX_VALUE);

    /**
     * Learning rate used to update the intercept (bias term).
     */
    public FloatOption biasLearningRateOption = new FloatOption(
            "biasLearningRate", 'b',
            "Learning rate for bias (intercept). If 0, the intercept is not updated.",
            0.01, 0.0, Double.MAX_VALUE);

    /**
     * L2 penalization hyper-parameter (weight decay).
     */
    public FloatOption l2Option = new FloatOption(
            "l2Penalty", 'l',
            "L2 regularization parameter (weight decay). Pushes weights towards 0.",
            0.0, 0.0, Double.MAX_VALUE);

    /**
     * Initial value for the bias (intercept).
     */
    public FloatOption initialBiasOption = new FloatOption(
            "initialBias", 'i',
            "Initial value for the bias (intercept).",
            0.0, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);

    // ---------------------------------------------------------------
    // Model state
    // ---------------------------------------------------------------

    /**
     * One weight array per class. Index k corresponds to class k.
     */
    protected double[][] weights;

    /**
     * One bias (intercept) per class. Index k corresponds to class k.
     */
    protected double[] biases;

    /**
     * Number of classes seen/supported.
     */
    protected int numClasses;

    /**
     * Cached learning rate hyperparameter.
     */
    protected double learningRate;

    /**
     * Cached L2 regularization hyperparameter.
     */
    protected double l2Regularization;

    // ---------------------------------------------------------------
    // Core methods
    // ---------------------------------------------------------------

    /**
     * Resets the model to its initial state.
     */
    @Override
    public void resetLearningImpl() {
        this.weights = null;
        this.biases = null;
        this.numClasses = 0;

        // Read hyperparameters
        this.learningRate = this.learningRateOption.getValue();
        this.l2Regularization = this.l2Option.getValue();
    }

    /**
     * Ensures internal structures are initialized and handle the required number of
     * classes.
     */
    protected void prepareForClasses(int requiredClasses) {
        if (this.numClasses >= requiredClasses) {
            return;
        }

        int newNumClasses = requiredClasses;
        double[][] newWeights = new double[newNumClasses][];
        double[] newBiases = new double[newNumClasses];

        // Copy existing state
        if (this.weights != null) {
            System.arraycopy(this.weights, 0, newWeights, 0, this.numClasses);
        }
        if (this.biases != null) {
            System.arraycopy(this.biases, 0, newBiases, 0, this.numClasses);
        }

        // Initialize new components
        double initialBias = this.initialBiasOption.getValue();
        for (int i = this.numClasses; i < newNumClasses; i++) {
            newWeights[i] = new double[0];
            newBiases[i] = initialBias;
        }

        this.weights = newWeights;
        this.biases = newBiases;
        this.numClasses = newNumClasses;
    }

    /**
     * Computes raw score (logit) for a specific class.
     */
    protected double computeLogit(Instance inst, int classIndex) {
        if (classIndex >= this.numClasses) {
            return 0.0;
        }

        double result = this.biases[classIndex];
        double[] classWeights = this.weights[classIndex];

        for (int i = 0; i < inst.numValues(); i++) {
            int idx = inst.index(i);
            if (idx != inst.classIndex() && !inst.isMissingSparse(i)) {
                if (idx < classWeights.length) {
                    result += inst.valueSparse(i) * classWeights[idx];
                }
            }
        }
        return result;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (inst.classIsMissing()) {
            return;
        }

        int targetClass = (int) inst.classValue();
        int instNumClasses = inst.numClasses();

        // Ensure we handle at least the number of classes defined in the
        // instance/target
        prepareForClasses(Math.max(instNumClasses, targetClass + 1));

        // 1. Forward pass: compute logits
        double[] logits = new double[this.numClasses];
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < this.numClasses; k++) {
            logits[k] = computeLogit(inst, k);
            if (logits[k] > maxLogit) {
                maxLogit = logits[k];
            }
        }

        // 2. Compute probabilities using Softmax (with max-subtraction stabilization)
        double[] probs = new double[this.numClasses];
        double sumExp = 0.0;
        for (int k = 0; k < this.numClasses; k++) {
            probs[k] = Math.exp(logits[k] - maxLogit);
            sumExp += probs[k];
        }
        for (int k = 0; k < this.numClasses; k++) {
            probs[k] /= sumExp;
        }

        // 3. Update weights and biases for each class
        double lrBias = this.biasLearningRateOption.getValue();

        for (int k = 0; k < this.numClasses; k++) {
            double target = (k == targetClass) ? 1.0 : 0.0;
            double lossGradient = (probs[k] - target) * inst.weight();

            // Gradient descent update of the bias (no regularization)
            this.biases[k] -= lrBias * lossGradient;

            double[] classWeights = this.weights[k];

            // Update feature weights
            for (int i = 0; i < inst.numValues(); i++) {
                int idx = inst.index(i);
                if (idx != inst.classIndex() && !inst.isMissingSparse(i)) {
                    // Resize weights array if necessary
                    if (idx >= classWeights.length) {
                        int newSize = Math.max(classWeights.length * 2, idx + 1);
                        double[] newW = new double[newSize];
                        System.arraycopy(classWeights, 0, newW, 0, classWeights.length);
                        classWeights = newW;
                        this.weights[k] = classWeights;
                    }

                    double xi = inst.valueSparse(i);
                    double currentWeight = classWeights[idx];

                    // L2 Regularization
                    if (this.l2Regularization > 0.0) {
                        currentWeight *= (1.0 - this.learningRate * this.l2Regularization);
                    }

                    // SGD Update
                    currentWeight -= this.learningRate * lossGradient * xi;
                    classWeights[idx] = currentWeight;
                }
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.numClasses == 0) {
            return new double[inst.numClasses()];
        }

        int requiredClasses = Math.max(inst.numClasses(), this.numClasses);
        // We don't call prepareForClasses during inference to avoid side effects,
        // but we handle potential missing class structures in computeLogit.

        double[] logits = new double[requiredClasses];
        double maxLogit = Double.NEGATIVE_INFINITY;

        for (int k = 0; k < requiredClasses; k++) {
            logits[k] = computeLogit(inst, k);
            if (logits[k] > maxLogit) {
                maxLogit = logits[k];
            }
        }

        double[] probs = new double[requiredClasses];
        double sumExp = 0.0;
        for (int k = 0; k < requiredClasses; k++) {
            probs[k] = Math.exp(logits[k] - maxLogit);
            sumExp += probs[k];
        }
        for (int k = 0; k < requiredClasses; k++) {
            probs[k] /= sumExp;
        }

        // If instance expects less classes than we have, return the expected number
        if (inst.numClasses() < requiredClasses) {
            double[] returnedProbs = new double[inst.numClasses()];
            System.arraycopy(probs, 0, returnedProbs, 0, inst.numClasses());
            return returnedProbs;
        }

        return probs;
    }

    @Override
    public void getModelDescription(StringBuilder result, int indent) {
        StringUtils.appendIndented(result, indent, toString());
        StringUtils.appendNewline(result);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[] {
                new Measurement("num classes", this.numClasses),
                new Measurement("num weights total", this.numClasses
                        * (this.weights != null && this.weights.length > 0 ? this.weights[0].length : 0)),
                new Measurement("biasLearningRate", this.biasLearningRateOption.getValue()),
                new Measurement("initialBias", this.initialBiasOption.getValue())
        };
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public String toString() {
        if (this.numClasses == 0) {
            return "SoftmaxRegression: No model built yet.\n";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("SoftmaxRegression (Cross-Entropy / Softmax)\n");
        sb.append("  Learning Rate: ").append(this.learningRate).append("\n");
        sb.append("  Bias Learning Rate: ").append(this.biasLearningRateOption.getValue()).append("\n");
        sb.append("  L2 Regularization: ").append(this.l2Regularization).append("\n");
        sb.append("  Initial Bias: ").append(this.initialBiasOption.getValue()).append("\n");
        sb.append("  Number of Classes: ").append(this.numClasses).append("\n");
        if (this.biases != null) {
            for (int k = 0; k < this.numClasses; k++) {
                sb.append("  Bias[" + k + "]: ").append(this.biases[k]).append("\n");
            }
        }
        return sb.toString();
    }
}
