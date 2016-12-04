import java.util.List;

/**
 * Classifier client
 *
 */
public class LogisticRegressionClassifierClient {
	
	public static final String TOY_TRAINING_DATA_FILE_PATH = "toy_data/a5a.train";
	public static final String TOY_TESTING_DATA_FILE_PATH = "toy_data/a5a.test";

	public static final String TRAINING_DATA_FILE_PATH = "data/a5a.train";
	public static final String TESTING_DATA_FILE_PATH = "data/a5a.test";
	
	public static void main(String[] args) {
		
		LogisticRegressionClassifierClient logisticRegressionClassifierClient = new LogisticRegressionClassifierClient();
		logisticRegressionClassifierClient.runClassifier();
		
	}
	
	private void runClassifier() {
		
		DataFileReader dataFileReader = new DataFileReader(TOY_TRAINING_DATA_FILE_PATH, TOY_TESTING_DATA_FILE_PATH);
		List<List<Double>> trainingDataFeatures = dataFileReader.getTrainingDataFeatures();
		List<BinaryDataLabel> trainingDataLabels = dataFileReader.getTrainingDataLabels();
		
		LogisticRegressionClassifier classifier = new LogisticRegressionClassifier();
		classifier.fit(trainingDataFeatures, trainingDataLabels);
		
		List<BinaryDataLabel> predictions = classifier.getPredictions(dataFileReader.getTestingDataFeatures());
		
		ClassifierMetrics classifierMetrics = new ClassifierMetrics(dataFileReader.getTestingDataLabels(), predictions);
		System.out.println("Accuracy: " + classifierMetrics.getAccuracy());
		
	}
}
