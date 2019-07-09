/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.bitex.bigdata.api.weka.sample;


import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
//import weka.classifiers.trees.RandomForest;
//import weka.classifiers.functions.MultilayerPerceptron;



/**
 *
 * @author HH
 */
public class ModelGenerator {
    public static <T> T  getBuilder(Class<T> clazz) throws IllegalAccessException, InstantiationException{
        
        T t = clazz.newInstance();
        return t;
    }
    
    public Classifier buildClassifier(String modelName,Instances traindataset)  throws IllegalAccessException, InstantiationException,ClassNotFoundException{
//        MultilayerPerceptron m = new MultilayerPerceptron();

//        Class clazz = Class.forName(model);
//        Classifier m = new ClassifierFactory().getBuilder(clazz);
//        Classifier m=new ClassifierFactory() .getBuilder(Class.forName(model));

        Class clazz =Class.forName(modelName);
        Classifier m = (Classifier)ModelGenerator.getBuilder(clazz);
        try {
            m.buildClassifier(traindataset);
        } catch (Exception ex) {
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
        return m;
    }
   
    public String evaluateModel(Classifier model, Instances traindataset, Instances testdataset) {
        Evaluation eval = null;
        try {
            // Evaluate classifier with test dataset
            eval = new Evaluation(traindataset);
            eval.evaluateModel(model, testdataset);
        } catch (Exception ex) {
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
        return eval.toSummaryString("", true);
    }
    
    public String evaluateModelCrossValidation(String modelName, Instances alldataset)  throws IllegalAccessException, InstantiationException,ClassNotFoundException{
        Evaluation eval=null;
        Class clazz =Class.forName(modelName);
        Classifier model = (Classifier)ModelGenerator.getBuilder(clazz);
        try {
            // Evaluate classifier with test dataset
            eval = new Evaluation(alldataset);
            eval.crossValidateModel(model, alldataset, 10, new Random(1));
    
        } catch (Exception ex) {
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
        return eval.toSummaryString("", true);
    }

    public void saveModel(Classifier model, String modelpath) {

        try {
            SerializationHelper.write(modelpath, model);
        } catch (Exception ex) {
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
