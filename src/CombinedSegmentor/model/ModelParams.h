#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	//neural parameters
	Alphabet embeded_words; // words
	LookupTable word_table; // should be initialized outside	
	Alphabet embeded_chars; // chars
	LookupTable char_table; // should be initialized outside	
	Alphabet embeded_chartypes; // chars
	LookupTable chartype_table; // should be initialized outside	
	Alphabet embeded_bichars; // bichars
	LookupTable bichar_table; // should be initialized outside
	Alphabet embeded_actions; // bichars
	LookupTable action_table; // should be initialized outside
	
	UniParams char_tanh_conv; // hidden
	LSTM1Params char_left_lstm; //left lstm
	LSTM1Params char_right_lstm; //right lstm
	BiParams word_conv;
	LSTM1Params word_lstm;
	BiParams action_conv;
	LSTM1Params action_lstm;
	FourParams sep_hidden;
	TriParams app_hidden;
	UniParams sep_score;
	UniParams app_score;
	
	//should be initialized outside
	Alphabet words; // words
	Alphabet chars; // chars
	Alphabet charTypes; // char type
	
	//feature templates
	//append feature parameters
	SparseC2Params  app_1C_C0;
	SparseC2Params  app_1Wc0_C0;
	SparseC3Params  app_2CT_1CT_CT0;

	//separate feature parameters
	SparseC2Params  sep_1C_C0;
	SparseC2Params  sep_1Wc0_C0;
	SparseC3Params  sep_2CT_1CT_CT0;
	SparseC1Params  sep_1W;
	SparseC2Params  sep_1WD_1WL;
	SparseC1Params  sep_1WSingle;
	SparseC2Params  sep_1W_C0;
	SparseC2Params  sep_2W_1W;
	SparseC2Params  sep_2Wc0_1W;
	SparseC2Params  sep_2Wcn_1W;
	SparseC2Params  sep_2Wc0_1Wc0;
	SparseC2Params  sep_2Wcn_1Wcn;
	SparseC2Params  sep_2W_1WL;
	SparseC2Params  sep_2WL_1W;
	SparseC2Params  sep_2W_1Wcn;
	SparseC2Params  sep_1Wc0_1WL;
	SparseC2Params  sep_1Wcn_1WL;
	SparseC2Params  sep_1Wci_1Wcn;


public:
	bool initial(HyperParams& opts){
		// some model parameters should be initialized outside
		if (words.size() <= 0 || chars.size() <= 0){
			return false;
		}
		//neural features
		char_tanh_conv.initial(opts.char_hidden_dim, opts.char_window_dim, true);
		char_left_lstm.initial(opts.char_lstm_dim, opts.char_hidden_dim); //left lstm
		char_right_lstm.initial(opts.char_lstm_dim, opts.char_hidden_dim); //right lstm
		word_conv.initial(opts.word_hidden_dim, opts.word_dim, opts.word_dim, true);
		word_lstm.initial(opts.word_lstm_dim, opts.word_hidden_dim);
		action_conv.initial(opts.action_hidden_dim, opts.action_dim, opts.action_dim, true);
		action_lstm.initial(opts.action_lstm_dim, opts.action_hidden_dim);
		sep_hidden.initial(opts.sep_hidden_dim, opts.char_lstm_dim, opts.char_lstm_dim, opts.word_lstm_dim, opts.action_lstm_dim, true);
		app_hidden.initial(opts.app_hidden_dim, opts.char_lstm_dim, opts.char_lstm_dim, opts.action_lstm_dim, true);
		sep_score.initial(1, opts.sep_hidden_dim, false);
		app_score.initial(1, opts.app_hidden_dim, false);

		//sparse features
		app_1C_C0.initial();
		app_1Wc0_C0.initial();
		app_2CT_1CT_CT0.initial();

		sep_1C_C0.initial();
		sep_1Wc0_C0.initial();
		sep_2CT_1CT_CT0.initial();
		sep_1W.initial();
		sep_1WD_1WL.initial();
		sep_1WSingle.initial();
		sep_1W_C0.initial();
		sep_2W_1W.initial();
		sep_2Wc0_1W.initial();
		sep_2Wcn_1W.initial();
		sep_2Wc0_1Wc0.initial();
		sep_2Wcn_1Wcn.initial();
		sep_2W_1WL.initial();
		sep_2WL_1W.initial();
		sep_2W_1Wcn.initial();
		sep_1Wc0_1WL.initial();
		sep_1Wcn_1WL.initial();
		sep_1Wci_1Wcn.initial();

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		//neural features
		word_table.exportAdaParams(ada);
		char_table.exportAdaParams(ada);
		chartype_table.exportAdaParams(ada);
		bichar_table.exportAdaParams(ada);
		action_table.exportAdaParams(ada);

		char_tanh_conv.exportAdaParams(ada);
		char_left_lstm.exportAdaParams(ada); 
		char_right_lstm.exportAdaParams(ada);
		word_conv.exportAdaParams(ada);
		word_lstm.exportAdaParams(ada);
		action_conv.exportAdaParams(ada);
		action_lstm.exportAdaParams(ada);
		sep_hidden.exportAdaParams(ada);
		app_hidden.exportAdaParams(ada);
		sep_score.exportAdaParams(ada);
		app_score.exportAdaParams(ada);

		//sparse features
		app_1C_C0.exportAdaParams(ada);
		app_1Wc0_C0.exportAdaParams(ada);
		app_2CT_1CT_CT0.exportAdaParams(ada);

		sep_1C_C0.exportAdaParams(ada);
		sep_1Wc0_C0.exportAdaParams(ada);
		sep_2CT_1CT_CT0.exportAdaParams(ada);
		sep_1W.exportAdaParams(ada);
		sep_1WD_1WL.exportAdaParams(ada);
		sep_1WSingle.exportAdaParams(ada);
		sep_1W_C0.exportAdaParams(ada);
		sep_2W_1W.exportAdaParams(ada);
		sep_2Wc0_1W.exportAdaParams(ada);
		sep_2Wcn_1W.exportAdaParams(ada);
		sep_2Wc0_1Wc0.exportAdaParams(ada);
		sep_2Wcn_1Wcn.exportAdaParams(ada);
		sep_2W_1WL.exportAdaParams(ada);
		sep_2WL_1W.exportAdaParams(ada);
		sep_2W_1Wcn.exportAdaParams(ada);
		sep_1Wc0_1WL.exportAdaParams(ada);
		sep_1Wcn_1WL.exportAdaParams(ada);
		sep_1Wci_1Wcn.exportAdaParams(ada);
	}

	//only for sparse features
	inline void setFixed(const int& base = 1){
		app_1C_C0.setFixed(base);
		app_1Wc0_C0.setFixed(base);
		app_2CT_1CT_CT0.setFixed(base);

		sep_1C_C0.setFixed(base);
		sep_1Wc0_C0.setFixed(base);
		sep_2CT_1CT_CT0.setFixed(base);
		sep_1W.setFixed(base);
		sep_1WD_1WL.setFixed(base);
		sep_1WSingle.setFixed(base);
		sep_1W_C0.setFixed(base);
		sep_2W_1W.setFixed(base);
		sep_2Wc0_1W.setFixed(base);
		sep_2Wcn_1W.setFixed(base);
		sep_2Wc0_1Wc0.setFixed(base);
		sep_2Wcn_1Wcn.setFixed(base);
		sep_2W_1WL.setFixed(base);
		sep_2WL_1W.setFixed(base);
		sep_2W_1Wcn.setFixed(base);
		sep_1Wc0_1WL.setFixed(base);
		sep_1Wcn_1WL.setFixed(base);
		sep_1Wci_1Wcn.setFixed(base);
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */