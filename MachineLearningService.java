package com.matt.service;

import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.jblas.DoubleMatrix;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.TreeSet;

@Service
public interface MachineLearningService {
    public Map<String, Object> trainingProcess(Map<Integer, List<String>> weatherMap, String city);

    public TreeSet<String> getCities(Map<Integer, List<String>> weatherData);

    public Map<String, Object> generatingTS(Map<Integer, List<String>> weatherData, String city);

    public DoubleMatrix getTsTrainingData(int col, Map<Integer, List<String>> map, String city);

    public HSSFWorkbook getExcel(Map<String, Object> predictions, Map<Integer, List<String>> weatherData, String city);
}
