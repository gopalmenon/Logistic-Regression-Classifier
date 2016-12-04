import java.util.Collections;
import java.util.List;

/**
 * Class to store feature and label vectors
 *
 */
public class FeaturesAndLabels {

	private List<List<Double>> features;
	private List<BinaryDataLabel> labels;
	
	/**
	 * Constructor
	 * @param features
	 * @param labels
	 */
	public FeaturesAndLabels(List<List<Double>> features, List<BinaryDataLabel> labels) {
		this.features = features;
		this.labels = labels;
	}

	public List<List<Double>> getFeatures() {
		return Collections.unmodifiableList(this.features);
	}

	public List<BinaryDataLabel> getLabels() {
		return Collections.unmodifiableList(this.labels);
	}

}
