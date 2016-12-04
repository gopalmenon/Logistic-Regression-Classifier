import java.util.List;

/**
 * Classifier client
 *
 */
public class LogisticRegressionClassifierClient {

	public static final String TRAINING_DATA_FILE_PATH = "data/a5a.train";
	public static final String TESTING_DATA_FILE_PATH = "data/a5a.test";
	
	public static void main(String[] args) {
		
		LogisticRegressionClassifierClient logisticRegressionClassifierClient = new LogisticRegressionClassifierClient();
		logisticRegressionClassifierClient.runClassifier();
		
	}
	
	private void runClassifier() {
		
		DataFileReader dataFileReader = new DataFileReader(TRAINING_DATA_FILE_PATH, TESTING_DATA_FILE_PATH);
		List<BinaryDataLabel> trainingDataLabels = dataFileReader.getTrainingDataLabels();
		List<List<Double>> trainingDataFeatures = dataFileReader.getTrainingDataFeatures();
		
		
	}
}
