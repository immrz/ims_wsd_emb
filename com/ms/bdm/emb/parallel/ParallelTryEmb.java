package com.ms.bdm.emb.parallel;

import sg.edu.nus.comp.nlp.ims.implement.*;
import sg.edu.nus.comp.nlp.ims.util.CScorer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.nio.file.Paths;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

public class ParallelTryEmb {
	
	static class Task {
		
		// the name of this task, to distinguish it from others
		protected String m_TaskName = null;
		
		// the embeddings to put into IMS
		protected String m_EmbFile = null;
		
		// this file may be used for testing, if attention strategy is adopted,
		// otherwise it is null.
		protected String m_SecondEmbFile = null;
		
		// the strategy to combine the embeddings
		protected String m_Strategy = null;
		
		// the directory where the datasets are put
		protected String m_DataDir = null;
		
		// the directory where the model is put
		protected String m_ModelDir = null;
		
		// the directory where the answers are put
		protected String m_AnsDir = null;
		
		protected String[] m_AlphaFile = null;
		
		protected static HashSet<String> m_NeedTwoEmb = new HashSet<String>(Arrays.asList(
			new String[] {"ATT", "SINGLE", "ONLYTGT"}
		));
		
		// some const parameters
		protected String m_ConstParamForTrain = "-prop E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\prop.xml"
				+ " -ptm E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\tag.bin.gz"
				+ " -tagdict E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\tagdict.txt"
				+ " -ssm E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\EnglishSD.bin.gz"
				+ " -f sg.edu.nus.comp.nlp.ims.feature.CFeatureExtractorCombination"
				+ " -s2 0"
				+ " -c2 0"
				+ " -ws 10";
		
		protected String m_ConstParamForTest = "-ptm E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\tag.bin.gz"
				+ " -tagdict E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\tagdict.txt"
				+ " -ssm E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\EnglishSD.bin.gz"
				+ " -prop E:\\users\\v-rumao\\codes\\ims_wsd_emb\\lib\\prop.xml"
				+ " -r sg.edu.nus.comp.nlp.ims.io.CResultWriter"
				+ " -f sg.edu.nus.comp.nlp.ims.feature.CFeatureExtractorCombination"
				+ " -ws 10";
		
		// all parameters
		protected ArrayList<String> m_Param = null;
		
		public static String embParentFolder = "E:\\users\\v-haoyal\\fromRunze\\data\\IMSContext";
		
		/**
		 * 
		 * @param taskName
		 * 			the id of this task
		 * @param embFile
		 * 			the absolute path of the embeddings file
		 * @param strategy
		 * 			AVG, EXP, ATT
		 * @param dataDir
		 * 			the absolute path of the directory where the datasets are located
		 * @param modelDir
		 * 			the absolute path of the directory where the model will be written
		 * @param ansDir
		 * 			the absolute path of the directory where the answers will be written
		 * @param secondEmb
		 * 			only needed when strategy is ATT, in which case embedding files
		 * 			for training and testing are different
		 */
		public Task(String taskName, String embFile, String strategy,
				String dataDir, String modelDir, String ansDir, String secondEmb) {
			
			this.m_TaskName = taskName;
			this.m_EmbFile = embFile;
			this.m_Strategy = strategy;
			this.m_DataDir = dataDir;
			this.m_ModelDir = modelDir;
			this.m_AnsDir = ansDir;
			
			if (Task.m_NeedTwoEmb.contains(strategy))
				this.m_SecondEmbFile = secondEmb;
		}
		
		/**
		 * prepare all the parameters for training
		 */
		private void prepareParamForTrain() {
			this.m_Param = new ArrayList<String>( Arrays.asList(this.m_ConstParamForTrain.split(" ")) );
			
			// specify embedding file
			this.m_Param.add("-emb");
			this.m_Param.add(this.m_EmbFile);
			
			// specify the strategy to use the embeddings
			this.m_Param.add("-str");
			this.m_Param.add(this.m_Strategy);
			
			// specify where the corpus is put
			this.m_Param.add("-readCorpus");
			this.m_Param.add(Paths.get(this.m_DataDir, "corpus.bin").toString());
			
			// specify where to read alpha features
			if (this.m_AlphaFile != null) {
				this.m_Param.add("-useAlpha");
				this.m_Param.add(this.m_AlphaFile[0]);
			}
			
			// specify the training file, key file and model directory
			this.m_Param.add(Paths.get(this.m_DataDir, "train.xml").toString());
			this.m_Param.add(Paths.get(this.m_DataDir, "train.key").toString());
			this.m_Param.add(this.m_ModelDir);
			
			System.out.println(this.m_TaskName + " training parameters:\n"
					+ String.join(" ", this.m_Param));
		}
		
		/**
		 * prepare all the parameters for testing
		 */
		private void prepareParamForTest() {
			this.m_Param = new ArrayList<String>( Arrays.asList(this.m_ConstParamForTest.split(" ")) );
			
			// specify embedding file
			this.m_Param.add("-emb");
			if (Task.m_NeedTwoEmb.contains(this.m_Strategy))
				this.m_Param.add(this.m_SecondEmbFile);
			else
				this.m_Param.add(this.m_EmbFile);
			
			// specify the strategy to use the embeddings
			this.m_Param.add("-str");
			this.m_Param.add(this.m_Strategy);
			
			// specify where the corpus is put
			this.m_Param.add("-readCorpus");
			this.m_Param.add(Paths.get(this.m_DataDir, "corpus_test.bin").toString());
			
			// specify where to read alpha features
			if (this.m_AlphaFile != null) {
				this.m_Param.add("-useAlpha");
				this.m_Param.add(this.m_AlphaFile[1]);
			}
			
			// specify the testing file, model directory,
			// statistic directory and answer directory
			this.m_Param.add(Paths.get(this.m_DataDir, "test.xml").toString());
			this.m_Param.add(this.m_ModelDir);
			this.m_Param.add(this.m_ModelDir);
			this.m_Param.add(this.m_AnsDir);
			
			System.out.println(this.m_TaskName + " testing parameters:\n"
					+ String.join(" ", this.m_Param));
		}
		
		/**
		 * train
		 */
		public void train() {
			this.prepareParamForTrain();
			CTrainModel.main(this.m_Param.toArray(new String[0]));
		}
		
		/**
		 * test
		 */
		public void test() {
			this.prepareParamForTest();
			CTester.main(this.m_Param.toArray(new String[0]));
		}
		
		/**
		 * merge the answers in answer directory
		 */
		public void mergeAnswers() throws Exception {
			ArrayList<String> answers = new ArrayList<String>();
			File ansDir = new File(this.m_AnsDir);
			for (File ans : ansDir.listFiles()) {
				String fileName = ans.getName();
				if (!ans.isFile() || !Character.isLetter(fileName.charAt(0)))
					continue;
				
				String item;
				if (this.m_TaskName.charAt(2) == '2')
					item = fileName.split("\\.")[0];
				else
					item = fileName.substring(0, fileName.lastIndexOf("."));
				
				// fix a small bug
				if (item.equals("colourless"))
					item = "colorless";
				
				try (BufferedReader reader = new BufferedReader(new FileReader(ans))) {
					String line;
					while ((line = reader.readLine()) != null) {
						String[] parts = line.split(" ");
						if (parts.length == 2)
							answers.add(item + " " + parts[0] + " " + parts[1] + "\n");
						else if (parts.length == 3)
							answers.add(item + " " + parts[1] + " " + parts[2] + "\n");
						else
							throw new Exception("wrong format answer: " + line);
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
			String outFileName = Paths.get(this.m_DataDir, this.m_TaskName + ".ans").toString();
			try (FileWriter writer = new FileWriter(outFileName)) {
				for (int i = 0; i < answers.size(); i++)
					writer.write(answers.get(i));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		/**
		 * call CScorer to score the answer
		 */
		public void score() {
			String answer = Paths.get(this.m_DataDir, this.m_TaskName + ".ans").toString();
			String key = Paths.get(this.m_DataDir, "test.key").toString();
			try {
				CScorer.main(new String[] {answer, key});
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public static Task build(String taskName, String datasetName,
				String[] embFileNames, String strategy) throws Exception {
	
			String imsParentFolder = "E:\\users\\v-rumao\\datasets\\IMS_NEW";
			
			String dataPath = Paths.get(imsParentFolder, datasetName, "dataDir").toString();
			String modelPath = Paths.get(imsParentFolder, datasetName, "modelDir").toString();
			String ansPath = Paths.get(imsParentFolder, datasetName, "ansDir").toString();
			
			String embFile, embFile2;
			if (Task.m_NeedTwoEmb.contains(strategy)) {
				
				if (embFileNames.length != 2)
					throw new Exception("Using attention strategy needs two embedding files.");
				
				embFile = embFileNames[0];
				embFile2 = embFileNames[1];
				
			} else {
				
				embFile = embFileNames[0];
				embFile2 = null;
			}
			
			return new Task(taskName, embFile, strategy,
							dataPath, modelPath, ansPath, embFile2);
		}
		
		public Task alpha(String[] alphaFile) {
			this.m_AlphaFile = alphaFile;
			return this;
		}
	}
	
	static class FeatPath {
		public static String[] buildPath(String parentFolder, String type, String obj, String version) {
			String post = "tokenized_obj_" + obj + "_v" + version + "_refine_100_300.txt";
			String[] paths = {Paths.get(parentFolder, type + "_train_" + post).toString(),
							  Paths.get(parentFolder, type + "_test_" + post).toString()};
			return paths;
		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		final String se2 = "se2-lex-sample";
		final String se3 = "se3-lex-sample";
		final String se7 = "se7-lex-sample";
		ArrayList<Task> tasks = new ArrayList<Task>();
		
		try {
//			tasks.add(Task.build(
//						"se2_refine100_300_iter3_negExp_O1N0_v7_alpha",
//						se2,
//						new String[] {"E:\\users\\v-rumao\\datasets\\english_lists\\vectors"
//								+ "\\syn1neg\\l2v_en_100_300_iter3_syn1neg.txt"},
//						"EXP"
//					).alpha(FeatPath.buildPath(
//							Paths.get(Task.embParentFolder, se2, "final").toString(),
//							"alphas", "sigmoid_cos", "7"))
//					);
			
			for (String dataset : new String[] {se2, se3, se7}) {
				String pre = dataset.split("-")[0];
				String curEmbFolder = Paths.get(Task.embParentFolder, dataset, "final").toString();
				
				tasks.add(Task.build(pre + "_45w_iter1_syn0",
						dataset,
						new String[] {"E:\\users\\v-rumao\\datasets\\english_lists\\vectors_NEW"
								+ "\\filtered_stopwords_45w_iter1.txt"},
						"EXP"));
				
				tasks.add(Task.build(pre + "_45w_iter1_neg",
						dataset,
						new String[] {"E:\\users\\v-rumao\\datasets\\english_lists\\vectors_NEW"
								+ "\\syn1neg\\filtered_stopwords_45w_iter1_syn1neg.txt"},
						"EXP"));
			}
			
			for (Task task : tasks) {
				task.train();
				task.test();
				task.mergeAnswers();
				task.score();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

}
