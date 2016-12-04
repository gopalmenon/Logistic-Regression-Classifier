import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Support Vector Machine implementation
 */
public class LogisticRegressionClassifier {

	public static final int DEFAULT_NUMBER_OF_EPOCHS = 20, DEFAULT_CROSS_VALIDATION_SPLITS = 5, NUMBER_OF_CROSS_VALIDATION_FOLDS = 6, MINIMUM_SHUFFLES = 100;
	public static final List<Double> DEFAULT_LEARNING_RATES = Arrays.asList(Math.pow(10.0, 0.0), Math.pow(10.0, -1.0), Math.pow(10.0, -2.0), Math.pow(10.0, -3.0), Math.pow(10.0, -4.0), Math.pow(10.0, -5.0), Math.pow(10.0, -6.0), Math.pow(10.0, -7.0), Math.pow(10.0, -8.0), Math.pow(10.0, -9.0), Math.pow(10.0, -10.0));
	public static final List<Double> DEFAULT_VARIANCE_VALUES = Arrays.asList(Math.pow(Math.pow(10.0, 0.0), 2.0), Math.pow(Math.pow(10.0, -1.0), 2.0), Math.pow(Math.pow(10.0, -2.0), 2.0), Math.pow(Math.pow(10.0, -3.0), 2.0), Math.pow(Math.pow(10.0, -4.0), 2.0), Math.pow(Math.pow(10.0, -5.0), 2.0), Math.pow(Math.pow(10.0, -6.0), 2.0), Math.pow(Math.pow(10.0, -7.0), 2.0), Math.pow(Math.pow(10.0, -8.0), 2.0), Math.pow(Math.pow(10.0, -9.0), 2.0), Math.pow(Math.pow(10.0, -10.0), 2.0));
	public static final String LOG_FILE_NAME = "LogFile.txt";
	
	private int numberOfEpochsForTraining;
	private int crossValidationSplits;
	private List<Double> learningRatesForTraining, varianceValuesForTraining;
	private List<Double> weightVector;
	private Random randomNumberGenerator;
	private boolean runInDebug;
	private PrintWriter out;
	private double currentLearningRate;
	private List<Double> svmObjectiveTrend;
	private List<Double> bestSvmObjectiveTrend;
	private int stochasticGradientDescentCounter;
	
	/**
	 * Constructor using default values
	 */
	public LogisticRegressionClassifier() {
		this(DEFAULT_NUMBER_OF_EPOCHS, DEFAULT_CROSS_VALIDATION_SPLITS, DEFAULT_LEARNING_RATES, DEFAULT_VARIANCE_VALUES, true, LOG_FILE_NAME);
	}
	
	/**
	 * Constructor
	 * @param numberOfEpochsForTraining
	 * @param crossValidationSplits
	 * @param learningRatesForTraining
	 * @param varianceValuesForTraining
	 */
	public LogisticRegressionClassifier(int numberOfEpochsForTraining, int crossValidationSplits, List<Double> learningRatesForTraining, List<Double> varianceValuesForTraining, boolean runInDebug, String logFileName) {
		
		this.numberOfEpochsForTraining = numberOfEpochsForTraining;
		this.crossValidationSplits = crossValidationSplits;
		this.learningRatesForTraining = learningRatesForTraining;
		this.varianceValuesForTraining = varianceValuesForTraining;
		this.weightVector = new ArrayList<Double>();
		this.randomNumberGenerator = new Random(0);
		this.runInDebug = runInDebug;
		try{
			this.out = new PrintWriter(new FileWriter(logFileName));
		} catch (IOException e) {
			System.err.println("IOException while opening file ");
			e.printStackTrace();
			System.exit(0);
		}
		this.svmObjectiveTrend = new ArrayList<Double>();
		this.bestSvmObjectiveTrend = new ArrayList<Double>();
		
	}
	
	/**
	 * Train the SVM
	 * @param featureVectors
	 * @param trainingDataLabels
	 */
	public void fit(List<List<Double>> featureVectors, List<BinaryDataLabel> trainingDataLabels) {
		
		double currentAccuracy = 0.0, maximumAccuracy = Double.MIN_VALUE;
				
		//Run through multiple learning rates
		for (Double learningRate : this.learningRatesForTraining) {
			
			List<Double> weightVector = null;
			
			//Run through multiple tradeoff values
			for (Double varianceValue : this.varianceValuesForTraining) {
				
				//Run k-fold cross validation
				double averageAccuracy = 0.0;
				
				List<FeaturesAndLabels> crossValidationData = getCrossValidationData(this.crossValidationSplits, featureVectors, trainingDataLabels);

				for (int crossValidationCounter = 0; crossValidationCounter < this.crossValidationSplits; ++crossValidationCounter) {
				
					//Load training and testing data
					List<List<Double>> trainingDataSubsetFeatures = new ArrayList<List<Double>>();
					List<BinaryDataLabel> trainingDataSubsetLabels = new ArrayList<BinaryDataLabel>();
					
					List<List<Double>> testingDataSubsetFeatures = new ArrayList<List<Double>>();
					List<BinaryDataLabel> testingDataSubsetLabels = new ArrayList<BinaryDataLabel>();
					
					int splitCounter = 0;
					for (FeaturesAndLabels featuresAndLabels : crossValidationData) {
					
						if (splitCounter == crossValidationCounter) {
							testingDataSubsetFeatures.addAll(featuresAndLabels.getFeatureVectors());
							testingDataSubsetLabels.addAll(featuresAndLabels.getLabels());
						} else {
							trainingDataSubsetFeatures.addAll(featuresAndLabels.getFeatureVectors());
							trainingDataSubsetLabels.addAll(featuresAndLabels.getLabels());
						}
					
						++splitCounter;
						
					}
					
					weightVector = getZeroWeightVector(trainingDataSubsetFeatures.get(1).size());
					
					//Run through multiple epochs
					boolean firstTime = true;
					for (int epochCounter = 0; epochCounter < this.numberOfEpochsForTraining; ++ epochCounter) {
						
						//Shuffle the training data for each subsequent epoch
						if (firstTime) {
							firstTime = false;
							this.currentLearningRate = learningRate.doubleValue();
							this.stochasticGradientDescentCounter = 0;
						} else {
							shuffleTrainingData(trainingDataSubsetFeatures, trainingDataSubsetLabels);
						}
						
						//Find the optimum weights by running stochastic gradient descent
						weightVector = runStochasticGradientDescent(trainingDataSubsetFeatures, trainingDataSubsetLabels, varianceValue.doubleValue(), weightVector);
						
						log("Log likelihood for learning rate: " + learningRate + ", variance value: " + varianceValue + ", epoch: " + epochCounter + " is " + getTotalLogLikelihood(trainingDataSubsetFeatures, trainingDataSubsetLabels, weightVector));
					
					}
					
					//Use the weight vector to run predictions
					List<BinaryDataLabel> predictions = getPredictions(testingDataSubsetFeatures, weightVector);
					
					//Get accuracy for current settings
					currentAccuracy = new ClassifierMetrics(testingDataSubsetLabels, predictions).getAccuracy();
					averageAccuracy += currentAccuracy;
					
				}
				
				//If this is the most accurate classification save the weight vector
				averageAccuracy /= this.crossValidationSplits;
				
				log("Learning rate: " + learningRate + ", variance value: " + varianceValue + ", average accuracy: " + averageAccuracy);
				
				if (averageAccuracy > maximumAccuracy) {
					maximumAccuracy = averageAccuracy;
					this.weightVector = weightVector;
					this.bestSvmObjectiveTrend = this.svmObjectiveTrend;
				}
			}
		
		}

	}
	
	/**
	 * @param stochasticGradientDescentCounter
	 * @param originalLearningRate
	 * @param varianceValue
	 * @return next learning rate
	 */
	private double getNextLearningRate(int stochasticGradientDescentCounter, double originalLearningRate, double varianceValue) {
		return originalLearningRate / (1 + (originalLearningRate * stochasticGradientDescentCounter / varianceValue));
	}
	
	/**
	 * @param trainingDataSubsetFeatures
	 * @param trainingDataSubsetLabels
	 * @param weightVector
	 * @return total log likelihood for the data set using the weight vector
	 */
	private double getTotalLogLikelihood(List<List<Double>> trainingDataSubsetFeatures, List<BinaryDataLabel> trainingDataSubsetLabels, List<Double> weightVector) {
		
		double totalLogLikelihood = 0.0;
		//Loop through each training record sample
		int featureVectorCounter = 0;
		for (List<Double> featureVector : trainingDataSubsetFeatures) {
			
			totalLogLikelihood += -1 * Math.log(1.0 + Math.pow(Math.E, -1.0 * trainingDataSubsetLabels.get(featureVectorCounter).getValue() * getDotProduct(weightVector, featureVector)));
			
			++featureVectorCounter;
		}
		
		return totalLogLikelihood;
	}
	
	/**
	 * @param testingData
	 * @return prediction labels
	 */
	public List<BinaryDataLabel> getPredictions(List<List<Double>> testingData) {
		
		List<BinaryDataLabel> predictionLabels = new ArrayList<BinaryDataLabel>(testingData.size());
		
		for (List<Double> testVector : testingData) {			
			predictionLabels.add(getPrediction(testVector));
		}
		
		return predictionLabels;
		
	}

	/**
	 * This method is used while doing cross-validation
	 * @param testingData
	 * @param weightVector
	 * @return predictions
	 */
	public List<BinaryDataLabel> getPredictions(List<List<Double>> testingData, List<Double> weightVector) {
		
		List<BinaryDataLabel> predictionLabels = new ArrayList<BinaryDataLabel>(testingData.size());
		
		for (List<Double> testVector : testingData) {
			if (getDotProduct(weightVector, adjustForBias(testVector)) >= 0) {
				predictionLabels.add(BinaryDataLabel.POSITIVE_LABEL);
			} else {
				predictionLabels.add(BinaryDataLabel.NEGATIVE_LABEL);
			}			
		}
		
		return predictionLabels;
		
	}	

	/**
	 * @param testVector
	 * @return prediction label
	 */
	public BinaryDataLabel getPrediction(List<Double> testVector) {
		
		if (getDotProduct(this.weightVector, adjustForBias(testVector)) >= 0) {
			return BinaryDataLabel.POSITIVE_LABEL;
		} else {
			return BinaryDataLabel.NEGATIVE_LABEL;
		}
		
	}
	
	/**
	 * @param featureVectorSize
	 * @return a zero weight vector
	 */
	private List<Double> getZeroWeightVector(int featureVectorSize) {
		
		List<Double> weightVector = new ArrayList<Double>();
		
		for (int weightVectorIndex = 0; weightVectorIndex < featureVectorSize; ++weightVectorIndex) {
			weightVector.add(Double.valueOf(0.0));
		}
		
		return weightVector;
		
	}
	
	/**
	 * @param inputVector
	 * @return vector with first term as 1 to account for the bias
	 */
	private List<Double> adjustForBias(List<Double> inputVector) {
		
		List<Double> vectorAdjustedForBias = new ArrayList<Double>(inputVector.size() + 1);
		
		vectorAdjustedForBias.add(1.0);
		for (Double feature : inputVector) {
			vectorAdjustedForBias.add(feature);
		}
		
		return vectorAdjustedForBias;
		
	}
	
	/**
	 * Shuffle the labels and features together
	 * @param featureVectors
	 * @param trainingDataLabels
	 */
	private void shuffleTrainingData(List<List<Double>> featureVectors, List<BinaryDataLabel> trainingDataLabels) {
		
		//Generate a random number for the number of times to shuffle the data  
		int numberOfTimesToSuffle = this.randomNumberGenerator.nextInt(MINIMUM_SHUFFLES + trainingDataLabels.size() / 2), swapContentsWith1 = 0, swapContentsWith2 = 0;
		BinaryDataLabel tempLabel;
		List<Double> tempFeatureVector = null;
		
		//Shuffle the data
		for (int shuffleCounter = 0; shuffleCounter < numberOfTimesToSuffle; ++shuffleCounter) {
			
			//Randomly generate the row numbers to shuffle
			swapContentsWith1 = this.randomNumberGenerator.nextInt(trainingDataLabels.size());
			swapContentsWith2 = this.randomNumberGenerator.nextInt(trainingDataLabels.size());
			
			//Swap the contents
			if (swapContentsWith1 != swapContentsWith2) {
				
				tempLabel = trainingDataLabels.get(swapContentsWith1);
				tempFeatureVector = featureVectors.get(swapContentsWith1);
				
				trainingDataLabels.set(swapContentsWith1, trainingDataLabels.get(swapContentsWith2));
				featureVectors.set(swapContentsWith1, featureVectors.get(swapContentsWith2));
				
				trainingDataLabels.set(swapContentsWith2, tempLabel);
				featureVectors.set(swapContentsWith2, tempFeatureVector);
				
			}
			
		}
		
	}
	
	/**
	 * @param numberOfCrossValidationFolds
	 * @param featuresVector
	 * @param labels
	 * @return list of features and labels lists
	 */
	private List<FeaturesAndLabels> getCrossValidationData(int numberOfCrossValidationFolds, List<List<Double>> featureVectors, List<BinaryDataLabel> labels) {
				
		List<FeaturesAndLabels> crossValidationData = new ArrayList<FeaturesAndLabels>(numberOfCrossValidationFolds);
		
		List<List<Double>> featuresVectorCopy = new ArrayList<List<Double>>(featureVectors);
		List<BinaryDataLabel> labelsCopy = new ArrayList<BinaryDataLabel>(labels);
		
		int numberOfCrossValidationDataRecords = labels.size() / numberOfCrossValidationFolds, randomRecordNumber = 0;
		
		//Create one less than the required number of splits
		for (int splitCounter = 0; splitCounter < numberOfCrossValidationFolds - 1; ++splitCounter) {

			List<List<Double>> featureVectorsSubset = new ArrayList<List<Double>>(numberOfCrossValidationDataRecords);
			List<BinaryDataLabel> labelsSubset = new ArrayList<BinaryDataLabel>(numberOfCrossValidationDataRecords);

			//Fill data required for cross validation split
			for (int recordCounter = 0; recordCounter < numberOfCrossValidationDataRecords; ++recordCounter) {
				
				randomRecordNumber = randomNumberGenerator.nextInt(labelsCopy.size());
				
				featureVectorsSubset.add(featuresVectorCopy.get(randomRecordNumber));
				featuresVectorCopy.remove(randomRecordNumber);
				
				labelsSubset.add(labelsCopy.get(randomRecordNumber));
				labelsCopy.remove(randomRecordNumber);

			}
			
			crossValidationData.add(new FeaturesAndLabels(featureVectorsSubset, labelsSubset));
			
		}
		
		//Add the remaining labels and features to the last split
		crossValidationData.add(new FeaturesAndLabels(featuresVectorCopy, labelsCopy));
		
		//Return the data
		return crossValidationData;
		
	} 
	
	/**
	 * @param trainingDataSubsetFeatures
	 * @param trainingDataSubsetLabels
	 * @param learningRate
	 * @param varianceValue
	 * @param weightVector
	 * @return
	 */
	private List<Double> runStochasticGradientDescent(List<List<Double>> trainingDataSubsetFeatures, List<BinaryDataLabel> trainingDataSubsetLabels, double varianceValue, List<Double> weightVector) {
		
		//Loop through each training record sample
		int featureVectorCounter = 0;
		for (List<Double> featureVector : trainingDataSubsetFeatures) {
		
			//Get depreciated learning rate
			this.currentLearningRate = getNextLearningRate(this.stochasticGradientDescentCounter++, this.currentLearningRate, varianceValue);
			
			//Update weight vector
			weightVector = getSumOfVectors(weightVector, multiplyWithVector(-1 * this.currentLearningRate, getSumOfVectors(multiplyWithVector(-1.0 * trainingDataSubsetLabels.get(featureVectorCounter).getValue() / (1.0 + Math.pow(Math.E, trainingDataSubsetLabels.get(featureVectorCounter).getValue() * getDotProduct(weightVector, featureVector))), featureVector), multiplyWithVector(2.0 / varianceValue, weightVector))));
			
			++featureVectorCounter;
			
		}
		
		return weightVector;
		
	}
	
	/**
	 * @param multiplyWith
	 * @param weightVector
	 * @return a vector that is the result of multiplying the input vector with a factor
	 */
	private List<Double> multiplyWithVector(double multiplyWith, List<Double> weightVector) {
		
		List<Double> newWeightVector = new ArrayList<Double>(weightVector.size());
		
		for (Double feature : weightVector) {
			newWeightVector.add(Double.valueOf(feature.doubleValue() * multiplyWith));
		}
		
		return newWeightVector;
		
	}
	
	/**
	 * @param vector1
	 * @param vector2
	 * @return the sum of two vectors
	 */
	private List<Double> getSumOfVectors(List<Double> vector1, List<Double> vector2) {
		
		assert vector1.size() == vector2.size();
		List<Double> sumVector = new ArrayList<Double>(weightVector.size());
		
		int vectorIndex = 0;
		for (Double vectorElement : vector1) {
			sumVector.add(Double.valueOf(vectorElement.doubleValue() + vector2.get(vectorIndex++).doubleValue()));
		}
		
		return sumVector;
		
	}
	
	/**
	 * @return the weight vector
	 */
	public List<Double> getWeightVector() {
		return this.weightVector;
	}
	
	public List<Double> getBestSvmObjectiveTrend() {
		return this.bestSvmObjectiveTrend;
	}
	
	/**
	 * Method that can be used for debugging. This will return the current value of the objective and this should reduce over time.
	 * 
	 * @param currentWeightVector
	 * @param currentFeatureVector
	 * @param currentFeatureLabel
	 * @return current value of the SVM Objective
	 */
	private double getCurrentSvmObjectiveValue(List<Double> currentWeightVector, List<Double> currentFeatureVector, BinaryDataLabel currentFeatureLabel, double currentVarianceValue) {
		
		if (this.runInDebug) {
			double currentSvmObjectiveValue = 0.5 * getDotProduct(currentWeightVector, currentWeightVector);
			
			currentSvmObjectiveValue += currentVarianceValue * Math.max(0.0, 1 - currentFeatureLabel.getValue() * getDotProduct(currentWeightVector, currentFeatureVector));
			
			this.svmObjectiveTrend.add(currentSvmObjectiveValue);
			
			return currentSvmObjectiveValue;
			
		} else {
			
			return Double.valueOf(0.0);
			
		}
	}
	
	/**
	 * Print the best SVM Objective trend
	 */
	public void printBestSvmObjectiveTrend() {
		
		this.out.print("c(");
		
		boolean firstTime = true;
		for (Double svmObjective : this.bestSvmObjectiveTrend) {
			
			if (firstTime) {
				firstTime = false;
			} else {
				this.out.print(", ");
			}
			
			this.out.print(svmObjective.toString());
		}
		
		this.out.print(")");
		
	}
	
	/**
	 * @param vector1
	 * @param vector2
	 * @return the dot product of two vectors
	 */
	private double getDotProduct(List<Double> vector1, List<Double> vector2) {
		
		//Both vectors need to have the same dimensions
		assert vector1.size() == vector2.size();
		
		//Compute the dot product
		double dotProduct = 0.0;
		int vectorIndex = 0;
		for (Double feature : vector1) {
			dotProduct += feature.doubleValue() * vector2.get(vectorIndex++).doubleValue();
		}
		return dotProduct;
		
	}

	/**
	 * Write log to console if running in debug
	 * 
	 * @param stringToLog
	 */
	private void log(String stringToLog) {
		
		if (this.runInDebug) {
			this.out.println(stringToLog);
		}
		
	}
	
	/**
	 * Close the log file
	 */
	public void closeLogFile() {
		this.out.close();
	}
	
}
