package application;

public class Main {
	
	public static void main(String[] args) throws Exception {
		String inputFilePath1 = args[0];
		String inputFilePath2 = args[1];
		String outputPath = args[2];
		
		Classification classeur = new Classification(inputFilePath1, inputFilePath2, outputPath);
		
		classeur.startClassification();
	}
}
