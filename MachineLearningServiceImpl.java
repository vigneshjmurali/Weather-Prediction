package com.matt.service.impl;

import com.matt.service.BaseService;
import com.matt.service.MachineLearningService;
import com.matt.util.MatrixUtil;
import com.matt.util.POIUtil;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.jblas.DoubleMatrix;
import org.springframework.stereotype.Service;

import java.text.DecimalFormat;
import java.util.*;

@Service
public class MachineLearningServiceImpl extends BaseService implements MachineLearningService {
    Map<String, DoubleMatrix> temperatureWieghts = new HashMap<>();
    Map<String, DoubleMatrix> dewPointWeights = new HashMap<>();
    int tempStep = 10;
    int tempStride = 36;
    int dewStep = 2;
    int dewStride = 1;

    @Override
    public Map<String, Object> trainingProcess(Map<Integer, List<String>> weatherMap, String city) {
        Map<String, Object> result;
        DoubleMatrix avgTempMat = getTsTrainingData(8, weatherMap, city);
        return timeSeriesProcess(avgTempMat, 200000, tempStep, tempStride, 8, 0.1, 0.8);
//        DoubleMatrix avgDewPtMat = getTsTrainingData(11, weatherMap, city);
//        dewPointWeights = timeSeriesProcess(avgDewPtMat, 80000, dewStep, dewStride, 3, 0.2, 0.9);
//        DoubleMatrix events = getTsTrainingData(25, weatherMap, city);
//        DoubleMatrix eventsInput = MatrixUtil.combiningCols(avgTempMat, avgDewPtMat);
//        mlp(eventsInput, events, 1000, 4, 0.2, eventWeights);
    }

    public Map<String, Object> timeSeriesProcess(DoubleMatrix originMat, int trainingTime, int step,
                                                 int stride, int hidden, double eta, double momentum){
        DoubleMatrix normalizedTemps = MatrixUtil.getNormalizedMatrix(originMat);
        DoubleMatrix tsMatrix = getTimeSeriesMatrix(normalizedTemps, step, stride);
        DoubleMatrix targets = tsMatrix.getColumn(step);
        DoubleMatrix tsMatrixB = MatrixUtil.getMatrixWithBias(
                tsMatrix.get(MatrixUtil.getIndicesArray(0, tsMatrix.getRows()), MatrixUtil.getIndicesArray(0, step))
        );
        DoubleMatrix v = DoubleMatrix.rand(tsMatrixB.getColumns(), hidden);
        DoubleMatrix temp = getTempForWeightSize(tsMatrixB, v);
        DoubleMatrix w = DoubleMatrix.rand(temp.getColumns(), targets.getColumns());
        Map<String, DoubleMatrix> weights = trainingTS(tsMatrixB, v, w, trainingTime, targets, eta, momentum);
        temperatureWieghts = weights;
        return testing(tsMatrixB, weights.get("v"), weights.get("w"), targets, originMat);
    }

    private Map<String, Object> testing(DoubleMatrix tsMatrixB, DoubleMatrix v, DoubleMatrix w,
                                              DoubleMatrix targets, DoubleMatrix avgTempMat) {
        DoubleMatrix hiddenLayer = tsMatrixB.mmul(v);
        passTransferFunction(hiddenLayer);
        DoubleMatrix hb = MatrixUtil.getMatrixWithBias(hiddenLayer);
        DoubleMatrix output = hb.mmul(w);
        output = MatrixUtil.getUnnormalizedMatrix(avgTempMat, output);
        targets = MatrixUtil.getUnnormalizedMatrix(avgTempMat, targets);
        DecimalFormat df=new DecimalFormat("#.#");
        log.info("distance : " + MatrixUtil.getDistance(output, targets));
        for(int i=0; i<output.getRows(); i++){
            log.info("\n" + df.format(output.getRow(i).get(0)) + " : " + targets.getRow(i).get(0));
        }
        Map<String, Object> result = new HashMap<>();
        result.put("output", output.elementsAsList());
        result.put("target", targets.elementsAsList());
        return result;
    }

    private void mlp(DoubleMatrix input, DoubleMatrix target, int trainingTime,
                     int hidden, double eta, Map<String, DoubleMatrix> weights){

        DoubleMatrix inputB = MatrixUtil.getMatrixWithBias(input);
        DoubleMatrix normalizedInput = MatrixUtil.getNormalizedMatrix(inputB);
        DoubleMatrix v = DoubleMatrix.rand(inputB.getColumns(), hidden);
        DoubleMatrix temp = getTempForWeightSize(inputB, v);
        DoubleMatrix w = DoubleMatrix.rand(temp.getColumns(), target.getColumns());
        weights.put("v", v);
        weights.put("w", w);
        trainingMLP(normalizedInput, target, weights, trainingTime, eta);
        testingMLP(normalizedInput, target, weights);
    }

    private DoubleMatrix mlpForwardProcess(DoubleMatrix inputB, Map<String, DoubleMatrix> weights){
        DoubleMatrix hiddenLayer = inputB.mmul(weights.get("v"));
        passTransferFunction(hiddenLayer);
        DoubleMatrix hb = MatrixUtil.getMatrixWithBias(hiddenLayer);
        DoubleMatrix output = hb.mmul(weights.get("w"));
        return output;
    }

    private void testingMLP(DoubleMatrix inputB, DoubleMatrix target, Map<String, DoubleMatrix> weights){
        DoubleMatrix output = mlpForwardProcess(inputB, weights);
        passTransferFunction(output);
        output.print();
        log.info(output.getRows() + " : " + target.getRows() + "\n\n");
        for(int r=0; r<output.getRows(); r++){
            log.info(output.getRow(r).get(0) + " : " + target.getRow(r).get(0));
        }
    }

    private void trainingMLP(DoubleMatrix inputB, DoubleMatrix target, Map<String, DoubleMatrix> weights, int trainingTime, double eta) {
        DoubleMatrix v = weights.get("v");
        DoubleMatrix w = weights.get("w");
        for(int i=0; i<trainingTime; i++){
            DoubleMatrix hiddenLayer = inputB.mmul(v);
            passTransferFunction(hiddenLayer);
            DoubleMatrix hb = MatrixUtil.getMatrixWithBias(hiddenLayer);
            DoubleMatrix output = hb.mmul(w);
            passTransferFunction(output);
            if(output.equals(target)){
                break;
            }
            DoubleMatrix one = new DoubleMatrix(output.getRows(), output.getColumns());
            MatrixUtil.initMatrix(1.0, one);
            DoubleMatrix dOutput = (output.sub(target)).mul(output).mul(one.sub(output));
            DoubleMatrix oneSubHb = DoubleMatrix.zeros(hb.getRows(), hb.getColumns()).add(1).sub(hb);
            DoubleMatrix dHb = hb.mul(oneSubHb).mul(dOutput.mmul(w.transpose()));
            w = w.sub(hb.transpose().mmul(dOutput).mul(eta));
            DoubleMatrix dh = dHb.get(MatrixUtil.getIndicesArray(0, dHb.getRows()),
                    MatrixUtil.getIndicesArray(0, dHb.getColumns()-1));
            v = v.sub(inputB.transpose().mmul(dh).mul(eta));
        }
        weights.put("v", v);
        weights.put("w", w);
    }


    private Map<String, DoubleMatrix> trainingTS(DoubleMatrix tsMatrixB, DoubleMatrix v, DoubleMatrix w, int trainingTime, DoubleMatrix targets, double eta, double momentum) {
        DoubleMatrix lastUpdatedV = v.dup();
        DoubleMatrix lastUpdatedW = w.dup();
        MatrixUtil.initMatrix(0.0, lastUpdatedV);
        MatrixUtil.initMatrix(0.0, lastUpdatedW);
        for(int i=0; i<trainingTime; i++){
            DoubleMatrix hiddenLayer = tsMatrixB.mmul(v);
            passTransferFunction(hiddenLayer);
            DoubleMatrix hb = MatrixUtil.getMatrixWithBias(hiddenLayer);
            DoubleMatrix output = hb.mmul(w);
            if(output.equals(targets)){
                break;
            }
            DoubleMatrix dOutput = output.sub(targets).mul(1.0/tsMatrixB.getRows());
            DoubleMatrix oneSubHb = DoubleMatrix.zeros(hb.getRows(), hb.getColumns()).add(1).sub(hb);
            DoubleMatrix dHb = hb.mul(oneSubHb).mul(dOutput.mmul(w.transpose()));
            lastUpdatedW = (hb.transpose().mmul(dOutput)).mul(eta).add(lastUpdatedW.mul(momentum));
            w = w.sub(lastUpdatedW);
            DoubleMatrix dh = dHb.get(MatrixUtil.getIndicesArray(0, dHb.getRows()),
                    MatrixUtil.getIndicesArray(0, dHb.getColumns()-1));
            lastUpdatedV = (tsMatrixB.transpose()).mmul(dh).mul(eta).add(lastUpdatedV.mul(momentum));
            v = v.sub(lastUpdatedV);
        }
        Map<String, DoubleMatrix> weights = new HashMap<>();
        weights.put("v", v);
        weights.put("w", w);
        return weights;
    }

    public DoubleMatrix getTsTrainingData(int col, Map<Integer, List<String>> map, String city){
        List<List<String>> avgTempList = new ArrayList<>();
        map.forEach((row, cols) -> {
            if(city.equals(cols.get(3))){
                avgTempList.add(map.get(row));
            }
        });
        DoubleMatrix trainingMatrix = new DoubleMatrix(avgTempList.size(), 1);
        for(int i=0; i<avgTempList.size(); i++){
            Double avg = Double.parseDouble(avgTempList.get(i).get(col));
            trainingMatrix.put(i, 0, avg);
        }
        return trainingMatrix;
    }



    private DoubleMatrix getTempForWeightSize(DoubleMatrix tsMatrixB, DoubleMatrix v) {
        DoubleMatrix temp = tsMatrixB.mmul(v);
        DoubleMatrix tempB = MatrixUtil.getMatrixWithBias(temp);
        return tempB;
    }



    private void passTransferFunction(DoubleMatrix mat) {
        for(int r=0; r<mat.getRows(); r++){
            for(int c=0; c<mat.getColumns(); c++){
                double val = mat.getRow(r).get(c);
                mat.put(r, c, (1.0 / (1.0 + Math.exp((-1.0)*val))));
            }
        }
    }

    private DoubleMatrix getTimeSeriesMatrix(DoubleMatrix matrix, int step, int stride) {
        DoubleMatrix tsMatrix = new DoubleMatrix(matrix.getRows() - step*stride, step+1);
        for(int r=0; r+step*stride < matrix.getRows(); r++){
            for(int c=0; c<=step; c++){
                tsMatrix.put(r, c, matrix.getRow(r+c*stride).get(0));
            }
        }
        return tsMatrix;
    }

    @Override
    public TreeSet<String> getCities(Map<Integer, List<String>> weatherData) {
        TreeSet<String> cities = new TreeSet<>();
        weatherData.forEach((row, cols) -> {
            cities.add(cols.get(3));
        });
        return cities;
    }

    @Override
    public Map<String, Object> generatingTS(Map<Integer, List<String>> weatherData, String city) {
        Map<String, Object> predictions = new HashMap<>();
        if(temperatureWieghts.size() != 0){
            predictions.put("temperature", new ArrayList<>());
//            predictions.put("dewPoint", new ArrayList<>());
            DoubleMatrix avgTempMat = getTsTrainingData(8, weatherData, city);
            DoubleMatrix avgDewPointMat = getTsTrainingData(11, weatherData, city);
            int numRows = avgTempMat.getRows();
            DoubleMatrix tempInput;
            DoubleMatrix dpInput;
            List<Double> predictedTemp = new ArrayList<>();
            for(int i=0; i<60; i++){
                tempInput = getPredictingTsMat(tempStep, tempStride, avgTempMat);
//                if(i !=0 && i % tempStride == 0){
//                    temperatureWieghts = timeSeriesProcess(avgTempMat, 50000, tempStep, tempStride, 2, 0.1, 0.9);
//                }
                Double result = tsPredicting(tempInput, temperatureWieghts);
                predictedTemp.add(result);
                avgTempMat = MatrixUtil.increaseRow(result, avgTempMat);
            }
            predictions.put("temperature", predictedTemp);
            /*
            for(int i=0; i<365; i++){
                dpInput = getPredictingTsMat(2, 90, avgDewPointMat);
                Double result = tsPredicting(dpInput, dewPointWeights);
                predictions.get("dewPoint").add(result);
                avgDewPointMat = MatrixUtil.increaseRow(result, avgDewPointMat);
//                dpInput = MatrixUtil.increaseRow(result, dpInput.getRows(MatrixUtil.getIndicesArray(1, 2)));
            }
            */
            return predictions;
        }else{
            return new HashMap<>();
        }
    }

    private DoubleMatrix getPredictingTsMat(int step, int stride, DoubleMatrix matrix) {
        DoubleMatrix newMat = new DoubleMatrix(step, 1);
        for(int i=0; i<step; i++){
            int lastRow = matrix.getRows()-1;
            double val = matrix.getRow(lastRow - i*stride).get(0);
            newMat.put(step-1-i, 0, val);
        }
        return newMat;
    }

    public HSSFWorkbook getExcel(Map<String, Object> predictions, Map<Integer, List<String>> weatherData, String city) {
        DoubleMatrix avgTempMat = getTsTrainingData(8, weatherData, city);
        avgTempMat.print();
        HSSFWorkbook workbook = new HSSFWorkbook();
        HSSFSheet sheet = workbook.createSheet("Prediction");
        POIUtil.setText(sheet, 0, 0, "temperatures");
        POIUtil.setText(sheet, 0, 1, "Original Sample");
        POIUtil.setText(sheet, 0, 2, "Dew Point");
        List<Double> temps = (List<Double>) predictions.get("temperature");
        for(int i=0; i<temps.size(); i++){
            POIUtil.setText(sheet, i+1, 0, String.valueOf(temps.get(i)));
            POIUtil.setText(sheet, i+1, 1, String.valueOf(avgTempMat.getRow(i).get(0)));
//            POIUtil.setText(sheet, i+1, 2, String.valueOf(predictions.get("dewPoint").get(i)));
        }
        return workbook;
    }

    private Double tsPredicting(DoubleMatrix input, Map<String, DoubleMatrix> weights) {
        DoubleMatrix inputB = MatrixUtil.getMatrixWithBias(MatrixUtil.getNormalizedMatrix(input).transpose());
        DoubleMatrix output = mlpForwardProcess(inputB, weights);
        output = MatrixUtil.getUnnormalizedMatrix(input, output);
        return output.getRow(0).get(0);
    }
}
