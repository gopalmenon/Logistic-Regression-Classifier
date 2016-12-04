import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Read train and test data files and parse data
 *
 */
public class DataFileReader {
	
	public static final String WHITESPACE_REGEX = "\\s";
	public static final String FEATURE_VALUE_SEPARATOR = ":";

	private String trainingDataFilePath, testingDataFilePath;
	private List<List<Double>> trainingDataFeatures, testingDataFeatures;
	private List<BinaryDataLabel> trainingDataLabels, testingDataLabels;
	private int maximumNumberOfFeatures;
	
	/**
	 * Constructor
	 * @param trainingDataFileName
	 * @param testingDataFileName
	 */
	public DataFileReader(String trainingDataFilePath, String testingDataFilePath) {
		this.trainingDataFilePath = trainingDataFilePath;
		this.testingDataFilePath = testingDataFilePath;
		this.maximumNumberOfFeatures = 0;
		parseData();
	}
	
	/**
	 * @return training data features
	 */
	public List<List<Double>> getTrainingDataFeatures() {
		return Collections.unmodifiableList(this.trainingDataFeatures);
	}
	
	/**
	 * @return training data labels
	 */
	public List<BinaryDataLabel> getTrainingDataLabels() {
		return Collections.unmodifiableList(this.trainingDataLabels);
	}
	
	/**
	 * @return testing data features
	 */
	public List<List<Double>> getTestingDataFeatures() {
		return Collections.unmodifiableList(this.testingDataFeatures);
	}
	
	/**
	 * @return testing data labels
	 */
	public List<BinaryDataLabel> getTestingDataLabels() {
		return Collections.unmodifiableList(this.testingDataLabels);
	}
	
	/**
	 * Parse data and load internal data structures
	 */
	private void parseData() {
				
		//Get training and testing data from files
		List<String> rawTrainingData = null, rawTestingData = null;
		try {
			
			rawTrainingData = getDataFileContents(this.trainingDataFilePath);
			rawTestingData = getDataFileContents(this.testingDataFilePath);
			
			//Set the maximum number of features
			setMaximumNumberOfFeatures(rawTrainingData, rawTestingData);
			
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
		
		this.trainingDataLabels = getDataLabels(rawTrainingData);
		this.trainingDataFeatures = getDataFeatures(rawTrainingData);
		
		this.testingDataLabels = getDataLabels(rawTestingData);
		this.testingDataFeatures = getDataFeatures(rawTestingData);
		
	}

	/**
	 * Loop through training and testing data and save the value of the last column number
	 */
	private void setMaximumNumberOfFeatures(List<String> rawTrainingData, List<String> rawTestingData) {
		
		this.maximumNumberOfFeatures = Math.max(getMaximumNumberOfFeatures(rawTrainingData), getMaximumNumberOfFeatures(rawTrainingData));

	}
	
	/**
	 * @param rawData
	 * @return the maximum number of features in a data set
	 */
	private int getMaximumNumberOfFeatures(List<String> rawData) {
		
		int maximumNumberOfFeatures = 0, currentMaximumNumberOfFeatures = 0;
		String[] tokens = null, columnAndData = null;;
		
		for (String dataRow : rawData) {
		
			try {
				
				//Split around whitre spaces
				tokens = dataRow.split(WHITESPACE_REGEX);
				
				//Column number and data are separated by a colon
				columnAndData = tokens[tokens.length - 1].split(FEATURE_VALUE_SEPARATOR);
				currentMaximumNumberOfFeatures = Integer.parseInt(columnAndData[0]);
				
				//Save maximum column number
				if (currentMaximumNumberOfFeatures > maximumNumberOfFeatures) {
					maximumNumberOfFeatures = currentMaximumNumberOfFeatures;
				}
				
			} catch (NumberFormatException e) {
				e.printStackTrace();
				System.exit(0);
			}
		
		}
		
		return maximumNumberOfFeatures;
	}
	
	/**
	 * @param filePath
	 * @return file contents as a list of strings
	 * @throws IOException
	 */
	private List<String> getDataFileContents(String filePath) throws IOException {
		
		List<String> fileContents = new ArrayList<String>();
		BufferedReader bufferedReader = null;
		
		try {
			bufferedReader = new BufferedReader(new FileReader(filePath));
			String fileLine = bufferedReader.readLine();
			
			while (fileLine != null) {
				
				if (fileLine.trim().length() > 0) {
					fileContents.add(fileLine.trim());
				}

				fileLine = bufferedReader.readLine();
			}
			
			bufferedReader.close();
			
		} catch (IOException e) {
			throw e;
		} catch (NumberFormatException e) {
			throw e;
		}
		
		return fileContents;

	}
	
	/**
	 * Get data features and set the value of the maximum number of features
	 * @param rawData
	 * @return
	 */
	private List<List<Double>> getDataFeatures(List<String> rawData) {
		
		List<List<Double>> dataFeatures = new ArrayList<List<Double>>(rawData.size());
		int currentFeatureNumber = 0, currentFeatureValue = 0, lastFeatureNumberAdded = 0;
		String[] features = null, featureComponents = null;
		boolean firstTime = true;
		
		//Loop through all the records
		for (String fileLine : rawData) {
			
			List<Double> featureValues = new ArrayList<Double>(this.maximumNumberOfFeatures + 1);
			
			//The first element will be 1 so that it gets multiplied by the bias in the weight vector
			featureValues.add(Double.valueOf(1.0));		
		
			//Split the feature vector into groups of feature numbers and feature values
			features = fileLine.split(DataFileReader.WHITESPACE_REGEX);
			
			//Extract the numeric feature values
			currentFeatureNumber = 0;
			currentFeatureValue = 0;
			lastFeatureNumberAdded = 0;	
			
			firstTime = true;
			
			for (String featureGroup : features) {
			
				if (firstTime) {
					firstTime = false;
					continue;
				}
				
				featureComponents = featureGroup.split(FEATURE_VALUE_SEPARATOR);
				
				try {
					currentFeatureNumber = Integer.parseInt(featureComponents[0]);
					currentFeatureValue = Integer.parseInt(featureComponents[1]);
				} catch (NumberFormatException e) {
					System.err.println("Invalid feature value found.");
					System.exit(0);
				}
				
				//Fill in the features values with zeros till the next value in the data record
				for (int featureCounter = lastFeatureNumberAdded + 1; featureCounter < currentFeatureNumber; ++featureCounter) {
					featureValues.add(Double.valueOf(0.0));
				}
				
				lastFeatureNumberAdded = currentFeatureNumber;
				
				//Fill in the next value from the data file
				featureValues.add(Double.valueOf(currentFeatureValue));
			}
			
			//Add the rest of the features with zero values
			for (int featureCounter = lastFeatureNumberAdded + 1; featureCounter <= this.maximumNumberOfFeatures; ++featureCounter) {
				featureValues.add(Double.valueOf(0.0));
			}
			
			dataFeatures.add(featureValues);
		
		}
		
		return dataFeatures;
		
	}
	
	/**
	 * Get data labels from first column of data
	 * @param rawData
	 * @return
	 */
	private List<BinaryDataLabel> getDataLabels(List<String> rawData) {
		
		List<BinaryDataLabel> dataLabels = new ArrayList<BinaryDataLabel>(rawData.size());
		
		//Loop through all the records
		for (String fileLine : rawData) {
			
			try {
				//Add the label to the list
				dataLabels.add(Integer.parseInt(fileLine.split(WHITESPACE_REGEX)[0]) == 1 ? BinaryDataLabel.POSITIVE_LABEL : BinaryDataLabel.NEGATIVE_LABEL);
			} catch (NumberFormatException e) {
				e.printStackTrace();
				System.exit(0);
			}
		}
		
		return dataLabels;
		
	}
}
