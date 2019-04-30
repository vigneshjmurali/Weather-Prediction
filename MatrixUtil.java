package com.matt.util;

import org.jblas.DoubleMatrix;

public class MatrixUtil {
    public static DoubleMatrix combiningCols(DoubleMatrix matA, DoubleMatrix matB){
        if(matA.getRows() != matB.getRows()){
            return new DoubleMatrix();
        }
        DoubleMatrix newMat = new DoubleMatrix(matA.getRows(), matA.getColumns() + matB.getColumns());
        for(int i=0; i<matA.getColumns(); i++){
            newMat.putColumn(i, matA.getColumn(i));
        }
        for(int i=0; i<matB.getColumns(); i++){
            newMat.putColumn(i + matA.getColumns(), matB.getColumn(i));
        }
        return newMat;
    }

    public static DoubleMatrix combiningRows(DoubleMatrix matA, DoubleMatrix matB){
        if(matA.getColumns() != matB.getColumns()){
            return new DoubleMatrix();
        }
        DoubleMatrix newMat = new DoubleMatrix(matA.getRows()+matB.getRows(), matA.getColumns());
        for(int i=0; i<matA.getRows(); i++){
            newMat.putRow(i, matA.getRow(i));
        }
        for(int i=0; i<matB.getRows(); i++){
            newMat.putRow(i + matA.getRows(), matB.getRow(i));
        }
        return newMat;
    }

    public static double getDistance(DoubleMatrix matA, DoubleMatrix matB){
        double sum = 0;
        for(int r=0; r<matA.getRows(); r++){
            for(int c=0; c<matA.getColumns(); c++){
                double temp;
                temp = matA.getRow(r).get(c) - matB.getRow(r).get(c);
                sum += temp * temp;
            }
        }
        return sum;
    }

    public static DoubleMatrix getNormalizedMatrix(DoubleMatrix originalMatrix){
        DoubleMatrix columnMaxs = originalMatrix.columnMaxs();
        DoubleMatrix columnMins = originalMatrix.columnMins();
        DoubleMatrix normalizedMatrix = originalMatrix.dup();
        for(int c=0; c<originalMatrix.getColumns(); c++){
            double min = columnMins.getRow(0).get(c);
            double max = columnMaxs.getRow(0).get(c);
            if(min != max){
               for(int r=0; r<originalMatrix.getRows(); r++){
                   Double originalValue = originalMatrix.getRow(r).get(c);
                   originalValue = (originalValue - min) / (max - min);
                   normalizedMatrix.put(r, c, originalValue);
               }
            }
        }
        return normalizedMatrix;
    }

    public static DoubleMatrix getUnnormalizedMatrix(DoubleMatrix origin, DoubleMatrix newMat){
        DoubleMatrix columnMaxs = origin.columnMaxs();
        DoubleMatrix columnMins = origin.columnMins();
        DoubleMatrix unNormalizedMatrix = newMat.dup();
        for(int c=0; c<origin.getColumns(); c++){
            double min = columnMins.getRow(0).get(c);
            double max = columnMaxs.getRow(0).get(c);
            if(min != max) {
                for (int r = 0; r < newMat.getRows(); r++) {
                    Double value = newMat.getRow(r).get(c) * (max - min) + min;
                    unNormalizedMatrix.put(r, c, value);
                }
            }
        }
        return unNormalizedMatrix;
    }

    public static DoubleMatrix getMatrixWithBias(DoubleMatrix origin) {
        DoubleMatrix newMatrix = new DoubleMatrix(origin.getRows(), origin.getColumns()+1);
        for(int c=0; c<origin.getColumns(); c++){
            newMatrix.putColumn(c, origin.getColumn(c));
        }
        DoubleMatrix bias = new DoubleMatrix(origin.getRows(), 1);
        initMatrix(-1.0, bias);
        newMatrix.putColumn(newMatrix.getColumns()-1, bias);
        return newMatrix;
    }

    public static void initMatrix(Double value, DoubleMatrix matrix){
        for(int r=0; r<matrix.getRows(); r++){
            for(int c=0; c<matrix.getColumns(); c++){
                matrix.put(r, c, value);
            }
        }
    }

    public static int[] getIndicesArray(int start, int end){
        int[] indices = new int[end-start];
        for(int c=start; c<end; c++){
            indices[c-start] = c;
        }
        return indices;
    }

    public static DoubleMatrix increaseRow(Double value, DoubleMatrix matrix){
        DoubleMatrix newMat = new DoubleMatrix(matrix.getRows()+1, 1);
        for(int r=0; r<matrix.getRows(); r++){
            newMat.put(r, 0, matrix.get(r, 0));
        }
        newMat.put(newMat.getRows()-1, 0, value);
        return newMat;
    }
}
