/*
 * Driver.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3LDG.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "ComputionGraph.h"

class Driver {
public:
	Driver(size_t memsize) {
		_batch = 0;
	}

	~Driver() {
		_batch = 0;
	}

public:
	Graph _cg;
	vector<GraphBuilder>  _builder;
	ModelParams _modelparams;  // model parameters
	HyperParams _hyperparams;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update

	int _batch;

public:

	inline void initial() {
		if (!_hyperparams.bValid()) {
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams)) {
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_hyperparams.print();
		_builder.resize(_hyperparams.batch);

		for (int idx = 0; idx < _hyperparams.batch; idx++) {
			_builder[idx].initial(&_cg, _modelparams, _hyperparams);
		}
		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
		_batch = 0;
	}


public:
	dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
		_eval.reset();
		dtype cost = 0.0;
		int num = sentences.size();
		_cg.clearValue();
		for (int idx = 0; idx < num; idx++) {
			_builder[idx].forward(&sentences[idx], &goldACs[idx]);
			_eval.overall_label_count += goldACs[idx].size();
		}
		cost += loss_google(num);
		_cg.backward();
		return cost;
	}

	void decode(const std::vector<string>& sentence, vector<string>& result) {
		_cg.clearValue();
		_builder[0].forward(&sentence);
		predict(result);
	}

	void extractFeat(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
		int num = sentences.size();
		for (int idx = 0; idx < num; idx++) {
			_builder[idx].extractFeat(&sentences[idx], &goldACs[idx]);
		}
	}

	void updateModel() {
		if (_ada._params.empty()) {
			_modelparams.exportModelParams(_ada);
		}
		//_ada.rescaleGrad(1.0 / _batch);
		_ada.update(10);
		//_ada.updateAdam(10);
		_batch = 0;
	}

	void writeModel();

	void loadModel();

private:
	// max-margin
	dtype loss(int num) {
		dtype cost = 0;
		for (int idx = 0; idx < num; idx++) {
			GraphBuilder* _pcg = &_builder[idx];
			int step = _pcg->outputs.size();
			_eval.correct_label_count += step;
			static PNode pBestNode = NULL;
			static PNode pGoldNode = NULL;
			static PNode pCurNode;

			int offset = _pcg->outputs[step - 1].size();
			for (int idx = 0; idx < offset; idx++) {
				pCurNode = _pcg->outputs[step - 1][idx].in;
				if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
					pBestNode = pCurNode;
				}
				if (_pcg->outputs[step - 1][idx].bGold) {
					pGoldNode = pCurNode;
				}
			}

			_batch++;

			if (pGoldNode != pBestNode) {
				pGoldNode->loss[0] = -1.0 / num;
				pBestNode->loss[0] = 1.0 / num;
				cost += 1.0;
			}
		}
		return cost;
	}

	dtype loss_google(int num) {
		dtype cost = 0.0;
		for (int idx = 0; idx < num; idx++) {
			GraphBuilder* _pcg = &_builder[idx];
			int maxstep = _pcg->outputs.size();
			if (maxstep == 0) cost += 1.0;
			_eval.correct_label_count += maxstep;
			static PNode pBestNode = NULL;
			static PNode pGoldNode = NULL;
			static PNode pCurNode;
			static dtype sum, max;
			static int curcount, goldIndex;
			static vector<dtype> scores;

			for (int step = maxstep - 1; step < maxstep; step++) {
				curcount = _pcg->outputs[step].size();
				max = 0.0;
				goldIndex = -1;
				pBestNode = pGoldNode = NULL;
				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = _pcg->outputs[step][idx].in;
					if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
						pBestNode = pCurNode;
					}
					if (_pcg->outputs[step][idx].bGold) {
						pGoldNode = pCurNode;
						goldIndex = idx;
					}
				}

				if (goldIndex == -1) {
					std::cout << "impossible" << std::endl;
				}
				pGoldNode->loss[0] = -1.0 / num;

				max = pBestNode->val[0];
				sum = 0.0;
				scores.resize(curcount);
				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = _pcg->outputs[step][idx].in;
					scores[idx] = exp(pCurNode->val[0] - max);
					sum += scores[idx];
				}

				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = _pcg->outputs[step][idx].in;
					pCurNode->loss[0] += scores[idx] / (sum * num);
				}

				cost += -log(scores[goldIndex] / sum);

				if (std::isnan(cost)) {
					std::cout << "debug" << std::endl;
				}

				_batch++;

			}
		}
		return cost;
	}


	void predict(vector<string>& result) {
		int step = _builder[0].outputs.size();
		_builder[0].states[step - 1][0].getSegResults(result);
	}

public:
	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
