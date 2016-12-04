import java.util.List;

/**
 *This class will store labels and features as separate lists
 *
 */
public class FeaturesAndLabels {

	private List<List<Double>> featureVectors;
	private List<BinaryDataLabel> labels;
	
	//Constructor
	public FeaturesAndLabels(List<List<Double>> featureVectors, List<BinaryDataLabel> labels) {
		this.featureVectors = featureVectors;
		this.labels = labels;
	}
	
	//Getters
	public List<BinaryDataLabel> getLabels() {
		return labels;
	}

	public List<List<Double>> getFeatureVectors() {
		return featureVectors;
	}

}