package com.ms.bdm.emb.strategy;

import com.ms.bdm.emb.CCtxEmbExtractor;
import sg.edu.nus.comp.nlp.ims.corpus.IItem;
import sg.edu.nus.comp.nlp.ims.feature.CDoubleFeature;
import sg.edu.nus.comp.nlp.ims.feature.IFeature;

public abstract class IntegrationStrategy {

	protected final int WINDOW;

	private IntegrationStrategy(int windowSize) {
		this.WINDOW = windowSize;
	}

	public abstract String getEmbeddingDimension(CCtxEmbExtractor extractor, int p_EmbeddingDimensionIndex);
	
	public abstract IFeature getNext(CCtxEmbExtractor extractor);
	
	protected String formEmbeddingDimensionName(int p_EmbeddingDimensionIndex) {
		return "DIM_" + p_EmbeddingDimensionIndex;
	}
	
	public static IntegrationStrategy exponential(int windowSize) {
		return new IntegrationStrategy(windowSize) {
			public String getEmbeddingDimension(CCtxEmbExtractor extractor, int p_EmbeddingDimensionIndex) {
				double val = 0.0;
				for (int i = extractor.getM_IndexInSentence() - WINDOW; i <= extractor.getM_IndexInSentence() + WINDOW; i++) {
					if (i > -1 && i < extractor.getSentenceSize()) {
						int relI = extractor.getM_IndexInSentence() - i;
						double coef = coeficient(relI);
						IItem item = extractor.getItem(i);
						String dim = item.get(0).toLowerCase();
						
						int instID = extractor.getInstanceIndex();
						if (extractor.getWordMap(instID).containsKey(dim)) {
							val += coef * extractor.getWordMap(instID).get(dim)[p_EmbeddingDimensionIndex];
						}
					}
				}
				return Double.toString(val);
			}

			private final double ALPHA = 1 - Math.pow(0.1, 1 / (WINDOW * 1.0));

			private double coeficient(int n) {
				if (n == 0)
					return 1;
				else
					return Math.pow(1 - ALPHA, Math.abs(n) - 1);
			}
			
			public IFeature getNext(CCtxEmbExtractor extractor) {
				IFeature feature = null;
				if (extractor.getM_EmbeddingDimensionIndex() >= 0 && extractor.getM_EmbeddingDimensionIndex() < extractor.getVectorSize()) {
					feature = new CDoubleFeature();
					
					feature.setKey(this.formEmbeddingDimensionName(extractor.getM_EmbeddingDimensionIndex()));
					feature.setValue(getEmbeddingDimension(extractor, extractor.getM_EmbeddingDimensionIndex()));
					extractor.setM_EmbeddingDimensionIndex(extractor.getM_EmbeddingDimensionIndex() + 1);
				}
				return feature;
			}

		};
	}
	
	public static IntegrationStrategy onlyTargetEmb(int windowSize) {
		return new IntegrationStrategy(windowSize) {
			
			public String getEmbeddingDimension(CCtxEmbExtractor extractor, int p_EmbeddingDimensionIndex) {
				int instID = extractor.getInstanceIndex();
				String dim = extractor.getItem(extractor.getM_IndexInSentence()).get(0).toLowerCase();
				double val = 0.0;
				if (extractor.getWordMap(instID).containsKey(dim)) {
					val = extractor.getWordMap(instID).get(dim)[p_EmbeddingDimensionIndex];
				}
				return Double.toString(val);
			}
			
			public IFeature getNext(CCtxEmbExtractor extractor) {
				IFeature feature = null;
				if (extractor.getM_EmbeddingDimensionIndex() >= 0 && extractor.getM_EmbeddingDimensionIndex() < extractor.getVectorSize()) {
					feature = new CDoubleFeature();
					
					feature.setKey(this.formEmbeddingDimensionName(extractor.getM_EmbeddingDimensionIndex()));
					feature.setValue(getEmbeddingDimension(extractor, extractor.getM_EmbeddingDimensionIndex()));
					extractor.setM_EmbeddingDimensionIndex(extractor.getM_EmbeddingDimensionIndex() + 1);
				}
				return feature;
			}
			
		};
	}
	
}
