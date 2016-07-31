package application;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

public class Classification {
	
	static public ArrayList<Integer> ssdListValue = new ArrayList<Integer>(Arrays.asList(25,26,30,45,46,47,48,50,51,52,55,57,58,59,60,61,62,63,64,65,66,73,74,98,111,113,115,121,122,123,127,132,134,135,136,137,139,140,145));
	static public ArrayList<Integer> mccListValue = new ArrayList<Integer>(Arrays.asList(1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21));
	 
	private String inputFilePath1;
	private String inputFilePath2;
	private String outputPath;
	
	private static String[] typeMusic = {"BIG_BAND", "BLUES_CONTEMPORARY", "COUNTRY_TRADITIONAL", "DANCE", "ELECTRONICA", 
			"EXPERIMENTAL", "FOLK_INTERNATIONAL", "GOSPEL", "GRUNGE_EMO", "HIP_HOP_RAP", "JAZZ_CLASSIC", "METAL_ALTERNATIVE", 
			"METAL_DEATH", "METAL_HEAVY", "POP_CONTEMPORARY", "POP_INDIE", "POP_LATIN", "PUNK", "REGGAE", "RNB_SOUL", 
			"ROCK_ALTERNATIVE", "ROCK_COLLEGE", "ROCK_CONTEMPORARY", "ROCK_HARD", "ROCK_NEO_PSYCHEDELIA"};
	
	Classification(String inputFilePath1, String inputFilePath2, String outputPath) {
		this.inputFilePath1 = inputFilePath1;
		this.inputFilePath2 = inputFilePath2;
		this.outputPath = outputPath;
	}
	
	private static String classToNominal(int cl)
	{
		return typeMusic[cl];
	}
	
	private Instances prepareData(String inputFilePath, Instances toClassify) throws Exception {
		Normalize normalize = new Normalize();
		String[] option = {"-S", "1.0",  "-T", "0.0"};
		normalize.setOptions(option);
		normalize.setInputFormat(toClassify);
		Filter.useFilter(toClassify,normalize);


		Remove rmTag = new Remove();
		String[] rmOptions = {"-R", "1,2"};
		rmTag.setOptions(rmOptions);
		rmTag.setInputFormat(toClassify);
		toClassify = Filter.useFilter(toClassify, rmTag);

		Remove rmUseless = new Remove();
		String[] options = new String[2];

		if(inputFilePath.contains("mfcc")){

			options[0] = "-R";

			rmUseless = new Remove();
			for (Integer i = 0; i < 18; i++)
			{
				if (!mccListValue.contains(i+1))
				{
					if(options[1] == null)
					{
						options[1] = Integer.toString(i+1);
					} 
					else 
					{
						options[1] = options[1] + "," + Integer.toString(i+1);
					}
				}
			}

		}
		else if (inputFilePath.contains("ssd")) 
		{

			options[0] = "-R";

			rmUseless = new Remove();
			for (Integer i = 0; i < 168; i++)
			{

				if (!ssdListValue.contains(i+1))
				{
					if(options[1] == null){
						options[1] = Integer.toString(i+1);
					} 
					else 
					{
						options[1] = options[1] + "," + Integer.toString(i+1);
					}
				}
			}
		} 
		rmUseless.setOptions(options);
		rmUseless.setInputFormat(toClassify);
		toClassify = Filter.useFilter(toClassify, rmUseless);
		return toClassify;
	}
	
	public  void startClassification() throws Exception{
		//Get the data to classify
		Instances toClassify1 = new Instances(new BufferedReader(new FileReader(inputFilePath1)));
		toClassify1.setClassIndex(toClassify1.numAttributes() - 1);
		
		Instances toClassify2 = new Instances(new BufferedReader(new FileReader(inputFilePath2)));
		toClassify2.setClassIndex(toClassify2.numAttributes() - 1);
		
		
		//Prepare the j48 classifier and classify according to it.
		ObjectInputStream adaboostSSD = new ObjectInputStream(getClass().getResourceAsStream("/models/adassdv2.model"));
		AdaBoostM1 adaboostSSDClassifier = (AdaBoostM1) adaboostSSD.readObject();

		ObjectInputStream adaboostMFCC = new ObjectInputStream(getClass().getResourceAsStream("/models/adamfcc5passj48.model"));
		AdaBoostM1 adaboostMFCCClassifier = (AdaBoostM1) adaboostMFCC.readObject();
		
		System.out.println("Filtering data...");
		toClassify1 = prepareData(inputFilePath1, toClassify1);
		toClassify2 = prepareData(inputFilePath2, toClassify2);
		
		adaboostSSD.close();
		adaboostMFCC.close();

		System.out.println("Classifying data...");
		classifyDataSet(toClassify1, toClassify2, adaboostSSDClassifier, adaboostMFCCClassifier, outputPath);
	}
	
	
	private static void classifyDataSet(Instances toClassify1, Instances toClassify2, Classifier classifierToUse1, Classifier classifierToUse2, String outputPath) throws Exception{
		
		PrintWriter output = new PrintWriter("./" + outputPath, "UTF-8");
		
		//Classify data according to the classifier passed to the method.
		for(int i = 0; i < toClassify1.numInstances(); i++){
			double[] classifier1 = classifierToUse1.distributionForInstance(toClassify1.get(i));
			double[] classifier2 = classifierToUse2.distributionForInstance(toClassify2.get(i));
			double bestPercent = 0;
			int bestIndex = 0;
			
			
			for(int j = 0; j < classifier1.length; j++)
			{
				if(classifier1[j] > bestPercent){
					bestPercent = classifier1[j];
					bestIndex = j;
				} else if (classifier2[j] > bestPercent){
					bestPercent = classifier2[j];
					bestIndex = j;
				}

			}
			output.println(classToNominal(bestIndex));
		}
		output.close();
	}
}
