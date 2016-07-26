package application;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

public class Classification {
	
	static public ArrayList<Integer> ssdListValue = new ArrayList<Integer>(Arrays.asList(25,26,30,45,46,47,48,50,51,52,55,57,58,59,60,61,62,63,64,65,66,73,74,98,111,113,115,121,122,123,127,132,134,135,136,137,139,140,145));
	static public ArrayList<Integer> mccListValue = new ArrayList<Integer>(Arrays.asList(1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21));
	static public ArrayList<Integer> derivateListValue = new ArrayList<Integer>(Arrays.asList(2,5,8,15,19,20,21,22,23,24,25,26,29,30,35,37,38,39,41,42,43,44,47,48,55,56,58,61,64,66,67,69,70,79,88,94));
	 
	private String inputFilePath1;
	private String inputFilePath2;
	private String outputPath;
	
	private static String[] typeMusic = {"BIG_BAND,BLUES_CONTEMPORARY","COUNTRY_TRADITIONAL","DANCE","ELECTRONICA","EXPERIMENTAL","FOLK_INTERNATIONAL"
			                     ,"GOSPEL","GRUNGE_EMO","HIP_HOP_RAP","JAZZ_CLASSIC","METAL_ALTERNATIVE","METAL_DEATH","METAL_HEAVY","POP_CONTEMPORARY"
			                     ,"POP_INDIE","POP_LATIN","PUNK,REGGAE","RNB_SOUL","ROCK_ALTERNATIVE","ROCK_COLLEGE","ROCK_CONTEMPORARY","ROCK_HARD"
			                     ,"ROCK_NEO_PSYCHEDELIA"};
	
	Classification(String inputFilePath1, String inputFilePath2, String outputPath) {
		this.inputFilePath1 = inputFilePath1;
		this.inputFilePath2 = inputFilePath2;
		this.outputPath = outputPath;
	}
	
	private int classToNumeric(String cl)
	{
		for (int i = 1; i < 26; i++)
		{
			if (this.typeMusic[i-1].contains(cl))
				return i;
			
		}
		return 0;
	}
	
	private static String classToNominal(int cl)
	{
		return typeMusic[cl-1];
	}
	
	private void preTraitement(String inputFilePath, Instances toClassify) throws Exception
	{
		
		Normalize normalize = new Normalize();
		String[] option = {"-S", "1.0",  "-T", "0.0"};
		normalize.setOptions(option);
		normalize.setInputFormat(toClassify);
		Filter.useFilter(toClassify, normalize);
		
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
					if(options[1] == null){
						options[1] = Integer.toString(i+1);
					} else {
						options[1] = options[1] + "," + Integer.toString(i+1);
					}
				}
			}
			
		}
		else if (inputFilePath.contains("ssd")) {
		
			
			options[0] = "-R";
	
			rmUseless = new Remove();
			for (Integer i = 0; i < 168; i++)
			{
				
				if (!ssdListValue.contains(i+1))
				{
					if(options[1] == null){
						options[1] = Integer.toString(i+1);
					} else {
						options[1] = options[1] + "," + Integer.toString(i+1);
					}
				}
			}
		} 
		else if (inputFilePath.contains("deriv")) 
		{
			options[0] = "-R";
		
			rmUseless = new Remove();
			for (Integer i = 0; i < 96; i++)
			{
				
				if (!derivateListValue.contains(i+1))
				{
					if(options[1] == null){
						options[1] = Integer.toString(i+1);
					} else {
						options[1] = options[1] + "," + Integer.toString(i+1);
					}
				}
			}
		}
		
		
		rmUseless.setOptions(options);
		rmUseless.setInputFormat(toClassify);
		toClassify = Filter.useFilter(toClassify, rmUseless);
	}
	
	public  void startClassification() throws Exception{
		//Get the data to classify
		Instances toClassify1 = new Instances(new BufferedReader(new FileReader(inputFilePath1)));
		toClassify1.setClassIndex(toClassify1.numAttributes() - 1);
		
		Instances toClassify2 = new Instances(new BufferedReader(new FileReader(inputFilePath2)));
		toClassify2.setClassIndex(toClassify2.numAttributes() - 1);
		
		
		//Prepare the j48 classifier and classify according to it.
		ObjectInputStream j48InputStream = new ObjectInputStream(getClass().getResourceAsStream("/models/treessd.model"));
		Classifier j48Classifier = (Classifier) j48InputStream.readObject();
		
		ObjectInputStream knnInputStream = new ObjectInputStream(getClass().getResourceAsStream("/models/knn.model"));
		Classifier knnClassifier = (Classifier) knnInputStream.readObject();

		preTraitement(inputFilePath1, toClassify1);
		preTraitement(inputFilePath2, toClassify2);
		
		j48InputStream.close();
		knnInputStream.close();

		classifyDataSet(toClassify1, toClassify2, j48Classifier, knnClassifier, outputPath);
	}
	
	
	private static void classifyDataSet(Instances toClassify1, Instances toClassify2, Classifier classifierToUse1, Classifier classifierToUse2, String outputPath) throws Exception{
		
		PrintWriter output = new PrintWriter("./" + outputPath, "UTF-8");
		
		//Classify data according to the classifier passed to the method.
		for(int i = 0; i < toClassify1.numInstances(); i++){
			double[] classifier1 = classifierToUse1.distributionForInstance(toClassify1.get(i));
			double[] classifier2 = classifierToUse2.distributionForInstance(toClassify2.get(i));
			double sumTemp = 0;
			int indexMajority = 0;
			
			for(int j = 0; j < classifier1.length; j++)
			{
				double sum = classifier1[j] * 2 + classifier2[j];
				
				if (sum > sumTemp)
				{
					sumTemp = sum;
					indexMajority = j;
				}
				
			}
			
			output.println(classToNominal(indexMajority));
		}
		
		output.close();
		
	}
}
