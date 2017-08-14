class SPAddNode : public Node {
	vector<PNode> ins;
	int nSize;
	int dimId;
public:
	SPAddNode() : Node(){
		ins.clear();
		nSize = 0;
		dimId = -1;
		node_type = "spadd";
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		ins.clear();
		nSize = 0;
	}

public:
	// please better restrict col to 1
	void forward(Graph *cg, const vector<PNode>& x, const int& dim) {
		nSize = x.size();
		ins.clear();
		for (int i = 0; i < nSize; i++){
			ins.push_back(x[i]);
		}

		degree = 0;
		dimId = dim;
		for (int i = 0; i < nSize; i++) {
			ins[i]->addParent(this);
		}

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, const int& dim){
		ins.clear();
		ins.push_back(x1);
		nSize = 1;
		degree = 0;
		dimId = dim;
		ins[0]->addParent(this);
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, const int& dim){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		nSize = 2;

		degree = 0;
		dimId = dim;
		for(int i = 0; i < nSize; ++i) { 
			ins[i]->addParent(this);
		}

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, const int& dim){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		nSize = 3;

		degree = 0;
		dimId = dim;
		for(int i = 0; i < nSize; ++i) { 
			ins[i]->addParent(this);
		}

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, const int& dim){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		nSize = 4;

		degree = 0;
		dimId = dim;
		for(int i = 0; i < nSize; ++i) { 
			ins[i]->addParent(this);
		}

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, const int& dim){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		nSize = 5;

		degree = 0;
		dimId = dim;
		for(int i = 0; i < nSize; ++i) { 
			ins[i]->addParent(this);
		}

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6, const int& dim){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		ins.push_back(x6);
		nSize = 6;

		degree = 0;
		dimId = dim;
		for(int i = 0; i < nSize; ++i) { 
			ins[i]->addParent(this);
		}

		cg->addNode(this);
	}
public:
	inline PExecute generate(bool bTrain);

	// better to rewrite for deep understanding
	inline bool typeEqual(PNode other) {
		return Node::typeEqual(other);
	}

public:
	inline void compute(){
		int oDim;
		dtype sum = 0;
		for (int idx = 0; idx < nSize; idx++){
			oDim = ins[idx]->val.dim;
			if (oDim == 1){
				sum += ins[idx]->val[0];
			}
			else if (dimId < oDim){
				sum += ins[idx]->val[dimId];
			}
		}
		val[0] += sum;
	}

	void backward(){
		int oDim;
		for (int i = 0; i < nSize; i++){
			oDim = ins[i]->val.dim;
			if (oDim == 1){
				ins[i]->loss[0] += loss[0];
			}
			else if (dimId < oDim){
				ins[i]->loss[dimId] += loss[0];
			}
		}
	}

};

class SPAddExecute : public Execute {
public:
	bool bTrain;
public:
	inline void  forward() {
		int count = batch.size();
		//#pragma omp parallel for schedule(static,1)
		for (int idx = 0; idx < count; idx++) {
			SPAddNode* ptr = (SPAddNode*)batch[idx];
			ptr->compute();
			ptr->forward_drop(bTrain);
		}
	}

	inline void backward() {
		int count = batch.size();
		//#pragma omp parallel for schedule(static,1)
		for (int idx = 0; idx < count; idx++) {
			SPAddNode* ptr = (SPAddNode*)batch[idx];
			ptr->backward_drop();
			ptr->backward();
		}
	}
};

inline PExecute SPAddNode::generate(bool bTrain) {
	SPAddExecute* exec = new SPAddExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}