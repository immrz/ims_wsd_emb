package com.ms.bdm.emb;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;

import sg.edu.nus.comp.nlp.ims.corpus.ICorpus;
import sg.edu.nus.comp.nlp.ims.feature.IFeature;
import sg.edu.nus.comp.nlp.ims.feature.IFeatureExtractor;
import sg.edu.nus.comp.nlp.ims.feature.CDoubleFeature;

public class CListEmbProductExtractor implements IFeatureExtractor {
	
	public ArrayList<double[]> m_Products = null;
	public ICorpus m_Corpus = null;
	public int m_Index = -1;
	public int m_EmbeddingDimensionIndex = -1;
	public IFeature m_CurrentFeature = null;
	
	public CListEmbProductExtractor(String p_EmbFile) {
		this.m_Products = new ArrayList<double[]>();
		
		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(
						new FileInputStream(p_EmbFile))
				)) {
			
			String line;
			int numCtx = 0;
			
			while((line = br.readLine()) != null) {
				String[] arr = line.split(" ");
				if (arr.length == 1 && arr[0].isEmpty()) {
					this.m_Products.add(new double[0]);
					continue;
				}
				double[] vector = new double[arr.length];
				for (int i = 0; i < arr.length; i++) {
					vector[i] = Double.parseDouble(arr[i]);
				}
				this.m_Products.add(vector);
				numCtx++;
			}
			
			System.out.println("Have read " + numCtx + " single vectors.");
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	protected boolean validIndex(int p_Index) {
		if (this.m_Corpus != null && this.m_Corpus.size() > p_Index
				&& p_Index >= 0) {
			return true;
		}
		return false;
	}
	
	@Override
	public boolean setCurrentInstance(int p_Index) {
		if (this.validIndex(p_Index)) {
			this.m_Index = p_Index;
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
		this.m_Corpus = p_Corpus;
		return true;
	}
	
	@Override
	public boolean restart() {
		this.m_EmbeddingDimensionIndex = 0;
		this.m_CurrentFeature = null;
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
	
	@Override
	public IFeature next() {
		IFeature feature = null;
		if (this.hasNext()) {
			feature = this.m_CurrentFeature;
			this.m_CurrentFeature = null;
		}
		return feature;
	}
	
	public IFeature getNext() {
		IFeature feature = null;
		if (this.m_EmbeddingDimensionIndex >= 0
				&& this.m_EmbeddingDimensionIndex < this.m_Products.get(this.m_Index).length) {
			feature = new CDoubleFeature();
			feature.setKey("PROD_" + this.m_EmbeddingDimensionIndex);
			feature.setValue(Double.toString(
					this.m_Products.get(this.m_Index)[this.m_EmbeddingDimensionIndex]));
			this.m_EmbeddingDimensionIndex++;
		}
		return feature;
	}
}
