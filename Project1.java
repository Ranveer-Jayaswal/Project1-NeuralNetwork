import java.util.concurrent.ThreadLocalRandom;

public class Project1 {
    public static void main(String[] args) {
        double[][] data = {
            {2.9, 2.4, 2.1, 2.5},
            {-6.8, -16.2, -22.6, -32.2},
            {-1.7, 1.2, 1.9, -1.9},
            {10.6} 
        };

        double[] targets = {10, 10, 10, 10, 8, 6, 4, 2, 0, 0, 0, 0, 2, 4, 6, 8, 10, 10, 10, 10, 10};
        double[] inputs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

        double[][] output = optimizeParams(data, inputs, targets);

        System.out.println("a_1= " + String.format("%.2f", output[0][0]) + ", c_1= " + String.format("%.2f", output[1][0]));
        System.out.println("a_2= " + String.format("%.2f", output[0][1]) + ", c_2= " + String.format("%.2f", output[1][1]));
        System.out.println("a_3= " + String.format("%.2f", output[0][2]) + ", c_3= " + String.format("%.2f", output[1][2]));
        System.out.println("a_4= " + String.format("%.2f", output[0][3]) + ", c_4= " + String.format("%.2f", output[1][3]));
    }

    public static double calculateLoss(double[] weightsLayer1, double[] biasesLayer1, double[] weightsLayerOut, double biasLayerOut, double[] x, double[] y) {
        double totalLoss = 0;
        double[] predictions = new double[y.length];

        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < weightsLayer1.length; j++) {
                predictions[i] += weightsLayerOut[j] * reluActivation((weightsLayer1[j] * x[i]) + biasesLayer1[j]);
            }
            totalLoss += Math.pow(predictions[i] + biasLayerOut - y[i], 2);
        }
        return totalLoss;
    }

    public static double reluActivation(double val) {
        return val > 0 ? val : 0.001 * val;
    }

    public static double[][] optimizeParams(double[][] params, double[] x, double[] y) {
        double[] bLayer1 = params[1];
        double[] bLayer1Up = new double[params[1].length];
        double[] bLayer1Down = new double[params[1].length];

        double[] wLayer1 = params[0];
        double[] wLayer1Up = new double[params[0].length];
        double[] wLayer1Down = new double[params[0].length];

        double[] wLayerOut = params[2];
        double[] wLayerOutUp = new double[params[0].length];
        double[] wLayerOutDown = new double[params[0].length];
        double bLayerOut = params[3][0];
        double bLayerOutUp = 0;
        double bLayerOutDown = 0;

        double gradWeight = 1;
        double gradBias = 1;
        double gradOutWeight = 1;
        double gradOutBias = 1;

        double[] currentCost = new double[params[0].length];
        double[] previousCost = new double[params[0].length];
        double costThreshold = 1E-20;

        double lrWeight = 0.000001;
        double lrBias = 0.000001;
        double stopTolerance = 1E-11;

        int count = 0;
        int maxCount = 100000000;

        while (((Math.abs(gradWeight) > stopTolerance) || (Math.abs(gradBias) > stopTolerance)) && (count < maxCount)) {
            for (int i = 0; i < params[0].length; i++) {
                for (int j = 0; j < params[0].length; j++) {
                    if (j == i) {
                        wLayer1Up[j] = wLayer1[j] + lrWeight;
                        wLayer1Down[j] = wLayer1[j] - lrWeight;
                        bLayer1Up[j] = bLayer1[j] + lrBias;
                        bLayer1Down[j] = bLayer1[j] - lrBias;
                        wLayerOutUp[j] = wLayerOut[j] + lrWeight;
                        wLayerOutDown[j] = wLayerOut[j] - lrWeight;
                        bLayerOutUp = bLayerOut + lrBias;
                        bLayerOutDown = bLayerOut - lrBias;
                    } else {
                        wLayer1Up[j] = wLayer1[j];
                        wLayer1Down[j] = wLayer1[j];
                        bLayer1Up[j] = bLayer1[j];
                        bLayer1Down[j] = bLayer1[j];
                        wLayerOutUp[j] = wLayerOut[j];
                        wLayerOutDown[j] = wLayerOut[j];
                        bLayerOutUp = bLayerOut;
                        bLayerOutDown = bLayerOut;
                    }
                }

                gradOutWeight = calculateLoss(wLayer1, bLayer1, wLayerOutUp, bLayerOut, x, y) - calculateLoss(wLayer1, bLayer1, wLayerOutDown, bLayerOut, x, y);
                gradOutBias = calculateLoss(wLayer1, bLayer1, wLayerOut, bLayerOutUp, x, y) - calculateLoss(wLayer1, bLayer1, wLayerOut, bLayerOutDown, x, y);
                gradWeight = calculateLoss(wLayer1Up, bLayer1, wLayerOut, bLayerOut, x, y) - calculateLoss(wLayer1Down, bLayer1, wLayerOut, bLayerOut, x, y);
                gradBias = calculateLoss(wLayer1, bLayer1Up, wLayerOut, bLayerOut, x, y) - calculateLoss(wLayer1, bLayer1Down, wLayerOut, bLayerOut, x, y);

                wLayer1[i] = wLayer1[i] - (gradWeight / 2);
                bLayer1[i] = bLayer1[i] - (gradBias / 2);
                wLayerOut[i] = wLayerOut[i] - (gradOutWeight / 2);
                bLayerOut = bLayerOut - (gradOutBias / 2);

                previousCost[i] = currentCost[i];
                currentCost[i] = calculateLoss(wLayer1, bLayer1, wLayerOut, bLayerOut, x, y);

                if (count % 100000 == 0) {
                    System.out.printf("Iteration %d: Cost = %f wInL = %.6f bInL = %.6f wOutL = %.6f bOutL = %.6f%n", count + 1, currentCost[i], wLayer1[i], bLayer1[i], wLayerOut[i], bLayerOut);
                }

                if (Math.abs(previousCost[i] - currentCost[i]) < costThreshold) {
                    int randomValue = ThreadLocalRandom.current().nextInt(1, 20);
                    System.out.printf("Iter: %d LOCAL MINIMUM ISSUE, NUMBER: %d, Cost %.6f%n", count + 1, randomValue, currentCost[i]);
                    wLayer1[i] += randomValue;
                    bLayer1[i] += randomValue;
                }
            }
            count++;

            if (count >= maxCount)
                System.out.println("Weights and Biases Cannot converge");
        }
        System.out.printf("Final Cost = %f%n", currentCost[0]);

        double[][] result = new double[params[0].length][params[0].length];
        for (int i = 0; i < params[0].length; i++) {
            result[0][i] = wLayer1[i];
            result[1][i] = bLayer1[i];
            System.out.print("Final w: " + Math.round(wLayerOut[i]));
            System.out.println("  Final b: " + Math.round(bLayerOut));
        }

        return result;
    }
}