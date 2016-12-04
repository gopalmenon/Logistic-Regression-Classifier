import java.util.Iterator;
import java.util.List;

public class ClassifierMetrics {

	private double precision;
	private double recall;
	private double accuracy;
	private double f1Score;
	
	public ClassifierMetrics(List<BinaryDataLabel> actuals, List<BinaryDataLabel> predictions) {
		
		int numberActualLabels = actuals.size(), testDataRecordCounter = 0;
		int truePositives = 0, trueNegatives = 0, falsePositives = 0, falseNegatives = 0;
		assert numberActualLabels == predictions.size() && numberActualLabels > 0;
		BinaryDataLabel actualLabel = null;
		
		Iterator<BinaryDataLabel> actualsIterator = actuals.iterator();
		
		while (actualsIterator.hasNext()) {
			
			actualLabel = actualsIterator.next();
			if (actualLabel == BinaryDataLabel.POSITIVE_LABEL) {
				if (predictions.get(testDataRecordCounter++) == BinaryDataLabel.POSITIVE_LABEL) {
					++truePositives;
				} else {
					++falseNegatives;
				}
			} else {
				if (actualLabel == predictions.get(testDataRecordCounter++)) {
					++trueNegatives;
				} else {
					++falsePositives;
				}
			}
		}
		
		this.precision = (double) truePositives /(truePositives + falsePositives);
		this.recall = (double) truePositives /(truePositives + falseNegatives);
		this.accuracy = (double) (truePositives + trueNegatives) / predictions.size();
		this.f1Score = (double) (2 * this.precision * this.recall) / (this.precision + this.recall);

	}

	public double getPrecision() {
		return precision;
	}

	public double getRecall() {
		return recall;
	}

	public double getAccuracy() {
		return accuracy;
	}

	public double getF1Score() {
		return f1Score;
	}
	
}
