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
    public static String DATASETNAME = "iris.arff";
    public static String DATASETPATH = DATASETDIR +DATASETNAME;
    
    public static String MODElDIR = "D:\\Dataset\\model\\";
    
    public static void main(String[] args) throws Exception {
//     mix dataset        
        Instances dataset = new Dataset().loadDataset(DATASETPATH);
        dataset.randomize(new Debug.Random(5));
        
//     Normalize dataset
        Filter filter = new Normalize();
        filter.setInputFormat(dataset);
        Instances datasetnor = Filter.useFilter(dataset, filter);

//     seperate into train and test dataset        
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        Instances trainDataset = new Instances(datasetnor, 0, trainSize);
        Instances testDataset = new Instances(datasetnor, trainSize, testSize);

//     build classifier with train dataset
        ArrayList<String> modelSet = new ArrayList<String>();
        modelSet.add("weka.classifiers.trees.RandomForest");
        modelSet.add("weka.classifiers.lazy.IBk");
        modelSet.add("weka.classifiers.functions.SMO");
        modelSet.add("weka.classifiers.bayes.NaiveBayes");
//        modelSet.add("weka.classifiers.functions.MultilayerPerceptron");
        
        for(String modelClass:modelSet){
            String MODElNAME= modelClass.split("\\.")[modelClass.split("\\.").length -1];
            System.out.println("---------------------------------------------------------------------");
            System.out.println(MODElNAME);
            ModelGenerator mg = new ModelGenerator();
            Classifier model = (Classifier) mg.buildClassifier(trainDataset,modelClass);

            // Evaluate classifier with test dataset
            String evalsummary = mg.evaluateModel(model, trainDataset, testDataset);
    
            System.out.println("Evaluation: " + evalsummary);
            System.out.println("---------------------------------------------------------------------");

            //Save model

            String MODElPATH = MODElDIR + DATASETNAME.split(".arff")[0]+MODElNAME;
            mg.saveModel(model, MODElPATH);
        }
    }
    
}
