
/**
 * Enum to hold binary data labels
 */
public enum BinaryDataLabel {
	
	POSITIVE_LABEL (+1),
	NEGATIVE_LABEL(-1);
	
	private final int label;
	
	/**
	 * Constructor
	 * @param label
	 */
	BinaryDataLabel(int label) {
		assert label == +1 || label == -1;
		this.label = label;
	}
	
	/**
	 * @return value of label
	 */
	public int getValue() {
		return this.label;
	}
	
	@Override
	public String toString() {
		if (this.label == +1) {
			return "+";
		} else {
			return "-";
		}
	}
	
}
