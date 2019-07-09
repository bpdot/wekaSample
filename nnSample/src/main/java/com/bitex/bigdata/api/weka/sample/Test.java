/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.bitex.bigdata.api.weka.sample;


import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;


/**
 *
 * @author HH
 */
public class Test {
    public static String DATASETDIR = "D:\\Dataset\\datasets-UCI\\UCI\\";
    public static String DATASETNAME = "mushroom.arff";
    public static String DATASETPATH = DATASETDIR +DATASETNAME;
    
    public static String MODElDIR = "D:\\Dataset\\model\\";
    
    public static void main(String[] args) throws Exception {
        
        boolean flagNormalize = true;
//     mix dataset        
        Instances dataset = new Dataset().loadDataset(DATASETPATH);
        dataset.randomize(new Debug.Random(4));
        
//     Normalize dataset
        Instances datasetnor;
        if (flagNormalize) {
            Filter filter = new Normalize();
            filter.setInputFormat(dataset);
            datasetnor = Filter.useFilter(dataset, filter);
        } else {
            datasetnor = dataset;
        }

//     build classifier with train dataset
        ArrayList<String> modelSet = new ArrayList<String>();
        modelSet.add("weka.classifiers.trees.RandomForest");
        modelSet.add("weka.classifiers.meta.AdaBoostM1");
//        modelSet.add("weka.classifiers.meta.AdditiveRegression");
        modelSet.add("weka.classifiers.lazy.IBk");
        modelSet.add("weka.classifiers.functions.SMO");
        modelSet.add("weka.classifiers.bayes.NaiveBayes");
        modelSet.add("weka.classifiers.meta.LogitBoost");
//        modelSet.add("weka.classifiers.functions.MultilayerPerceptron");
        
        for(String modelClass:modelSet){
            String MODElNAME= modelClass.split("\\.")[modelClass.split("\\.").length -1];
            System.out.println("---------------------------------------------------------------------");
            System.out.println(MODElNAME);
            ModelGenerator mg = new ModelGenerator();
            
            // Evaluate classifier with crossvalidation
            String evalsummary = mg.evaluateModelCrossValidation(modelClass, datasetnor);
            System.out.println("Evaluation: " + evalsummary);
            System.out.println("---------------------------------------------------------------------");
                   
            //Save model
            Classifier model = (Classifier) mg.buildClassifier(modelClass,datasetnor);
            String MODElPATH = MODElDIR + DATASETNAME.split(".arff")[0]+MODElNAME;
            mg.saveModel(model, MODElPATH);
        }
    }
    
}
