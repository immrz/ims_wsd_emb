package com.ms.bdm.emb;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.io.IOException;

import com.ms.bdm.emb.strategy.IntegrationStrategy;
import sg.edu.nus.comp.nlp.ims.corpus.ICorpus;
import sg.edu.nus.comp.nlp.ims.corpus.IItem;
import sg.edu.nus.comp.nlp.ims.corpus.ISentence;
import sg.edu.nus.comp.nlp.ims.feature.CDoubleFeature;
import sg.edu.nus.comp.nlp.ims.feature.IFeature;
import sg.edu.nus.comp.nlp.ims.feature.IFeatureExtractor;

public class CCtxEmbExtractor implements IFeatureExtractor {

	// corpus to be extracted
	protected ICorpus m_Corpus = null;

	// index of current instance
	protected int m_Index = -1;
	
	// current sentence to process
	protected ISentence m_Sentence = null;
	
	// item index in current sentence
	protected int m_IndexInSentence;
	
	// item index in current sentence
	protected double[] m_actualVector = null;
	
	// item length
	protected int m_InstanceLength;

	// index of collocation feature
	protected int m_EmbeddingDimensionIndex = -1;
	
	// current feature
	protected IFeature m_CurrentFeature = null;
	
	protected ArrayList<HashMap<String, double[]>> wordMaps;
	
	protected int vectorSize = -1;

	private IntegrationStrategy strategy;

	/**
	 * constructor
	 */
	public CCtxEmbExtractor(String embeddingsFile, IntegrationStrategy strategy) {
		this.wordMaps = new ArrayList<HashMap<String, double[]>>();
		this.strategy = strategy;
		try {

			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(embeddingsFile)));
			
			String line;
			int numCtx = 0;

			while ((line = reader.readLine()) != null) {
				try {
					HashMap<String, double[]> wordMap = new HashMap<String, double[]>();
					final String actualLine = line.trim();
					int numWord = Integer.parseInt(actualLine);
					
					for (int i = 0; i < numWord; i++) {
						final String record = reader.readLine();
						if (record == null)
							throw new IOException("not enough words in context " + numCtx);
						String[] array = record.split(" ");
						double[] vector = new double[array.length - 1];
						for (int j = 0; j < array.length - 1; j++)
							vector[j] = Double.parseDouble(array[j + 1]);
						wordMap.put(array[0], vector);
					}
					
					this.wordMaps.add(wordMap);
					numCtx++;
					reader.readLine(); // an empty line
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			this.vectorSize = this.wordMaps.iterator().next().values().iterator().next().length;
			reader.close();
			System.out.println("Totally " + numCtx + " contexts are read.");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public int getInstanceIndex() {
		return this.m_Index;
	}
	
	public int getM_IndexInSentence() {
		return m_IndexInSentence;
	}
	
	public int getSentenceSize() {
		return this.m_Sentence.size();
	}
	
	public IItem getItem(int pos) {
		return this.m_Sentence.getItem(pos);
	}
	
	public Map<String, double[]> getWordMap(int instID) {
		return this.wordMaps.get(instID);
	}
	
	protected boolean validIndex(int p_Index) {
		if (this.m_Corpus != null && this.m_Corpus.size() > p_Index
				&& p_Index >= 0) {
			return true;
		}
		return false;
	}
	
	public int getVectorSize() {
		return vectorSize;
	}

	@Override
	public boolean setCurrentInstance(int p_Index) {
		if (this.validIndex(p_Index)) {
			this.m_Index = p_Index;
			this.m_Sentence = this.m_Corpus.getSentence(this.m_Corpus
					.getSentenceID(p_Index));
			this.m_IndexInSentence = this.m_Corpus.getIndexInSentence(p_Index);
			this.m_InstanceLength = this.m_Corpus.getLength(p_Index);
			this.restart();
			return true;
		}
		return false;
	}

	@Override
	public String getCurrentInstanceID() {
		if (this.validIndex(this.m_Index)) {
			return this.m_Corpus.getValue(this.m_Index, "id");
		}
		return null;
	}

	@Override
	public boolean setCorpus(ICorpus p_Corpus) {
		if (p_Corpus == null) {
			return false;
		}
		this.m_IndexInSentence = -1;
		this.m_Corpus = p_Corpus;
		return true;
	}

	@Override
	public boolean restart() {		
		this.m_EmbeddingDimensionIndex = 0;
		this.m_CurrentFeature = null;
		this.m_actualVector = null;
		return this.validIndex(this.m_Index);
	}

	@Override
	public boolean hasNext() {
		if (this.m_CurrentFeature != null) {
			return true;
		}
		if (this.validIndex(this.m_Index)) {
			this.m_CurrentFeature = this.getNext();
			if (this.m_CurrentFeature != null) {
				return true;
			}
		}
		return false;
	}
	

	public IFeature next() {
		IFeature feature = null;
		if (this.hasNext()) {
			feature = this.m_CurrentFeature;
			this.m_CurrentFeature = null;
		}
		return feature;
	}
	
	private IFeature getNext() {
		return strategy.getNext(this);
		
	}	
	
	public int getM_EmbeddingDimensionIndex() {
		return m_EmbeddingDimensionIndex;
	}

	public void setM_EmbeddingDimensionIndex(int m_EmbeddingDimensionIndex) {
		this.m_EmbeddingDimensionIndex = m_EmbeddingDimensionIndex;
	}
	
	protected String getEmbeddingDimension(int p_EmbeddingDimensionIndex) {
		return strategy.getEmbeddingDimension(this, p_EmbeddingDimensionIndex);
	}
	
}
