#ifndef ONLINENODE_H_
#define ONLINENODE_H_

#include <vector>
#include <queue>

#include "online_forests/data.h"
#include "online_forests/hyperparameters.h"
#include "online_forests/randomtest.h"
#include "online_forests/utilities.h"

using namespace std;

class OnlineNode {
public:
    OnlineNode() {
        m_isLeaf = true;
    }

    OnlineNode(const Hyperparameters &hp, const int &numClasses, const int &numFeatures, const vector<double> &minFeatRange,
            const vector<double> &maxFeatRange, const int &depth) :
        m_numClasses(&numClasses), m_numFeatures(&numFeatures), m_depth(depth), m_isLeaf(true), m_counter(0.0), m_label(-1),
                m_parentCounter(0.0), m_hp(&hp), m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {
        for (int i = 0; i < numClasses; i++) {
            m_labelStats.push_back(0.0);
        }

        // Creating random tests
        for (int i = 0; i < hp.numRandomTests; i++) {
            HyperplaneFeature test(numClasses, numFeatures, hp.numProjectionFeatures, minFeatRange, maxFeatRange);
            m_onlineTests.push_back(test);
        }
    }

    OnlineNode(const Hyperparameters &hp, const int &numClasses, const int &numFeatures, const vector<double> &minFeatRange,
            const vector<double> &maxFeatRange, const int &depth, const vector<double> &parentStats) :
        m_numClasses(&numClasses), m_numFeatures(&numFeatures), m_depth(depth), m_isLeaf(true), m_counter(0.0), m_label(-1),
                m_parentCounter(0.0), m_hp(&hp), m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {
        m_labelStats = parentStats;
        m_label = argmax(m_labelStats);
        m_parentCounter = sum(m_labelStats);

        // Creating random tests
        for (int i = 0; i < hp.numRandomTests; i++) {
            HyperplaneFeature test(numClasses, numFeatures, hp.numProjectionFeatures, minFeatRange, maxFeatRange);
            m_onlineTests.push_back(test);
        }
    }

    ~OnlineNode() {
        if (!m_isLeaf) {
            delete m_leftChildNode;
            delete m_rightChildNode;
        }
    }

    void update(Sample &sample);

    Result eval(Sample &sample) {
        if (m_isLeaf) {
            Result result;
            if (m_counter + m_parentCounter) {
                result.confidence = m_labelStats;
                scale(result.confidence, 1.0 / (m_counter + m_parentCounter));
                result.prediction = m_label;
            } else {
                for (int i = 0; i < *m_numClasses; i++) {
                    result.confidence.push_back(1.0 / *m_numClasses);
                }
                result.prediction = 0;
            }

            return result;
        } else {
            if (m_bestTest.eval(sample)) {
                return m_rightChildNode->eval(sample);
            } else {
                return m_leftChildNode->eval(sample);
            }
        }
    }

    void writeNode(OnlineNode *node, double counter, FILE *fp)
    {
      queue<OnlineNode *> que;
      que.push(node);
      while(!que.empty()){
        OnlineNode *cur = que.front();
        if (cur->m_isLeaf){
          fprintf(fp,"L");
          fprintf(fp," %d",cur->m_depth);
          fprintf(fp," %d",cur->m_isLeaf);
          fprintf(fp," %lf",cur->m_counter);
          fprintf(fp," %d",cur->m_label);
          fprintf(fp," %lf",cur->m_parentCounter);
          for (int i = 0; i < cur->m_labelStats.size(); i++) {
            fprintf(fp," %lf",cur->m_labelStats[i]);
          }
          for (int i = 0; i < cur->m_onlineTests.size(); i++) {
            cur->m_onlineTests[i].writeTest(fp);
          }
          fprintf(fp,"\n");
          que.pop();
        } else {
          fprintf(fp,"S");
          fprintf(fp," %d",cur->m_depth);
          fprintf(fp," %d",cur->m_isLeaf);
          fprintf(fp," %lf",cur->m_counter);
          fprintf(fp," %d",cur->m_label);
          fprintf(fp," %lf",cur->m_parentCounter);
          for (int i = 0; i < cur->m_labelStats.size(); i++) {
            fprintf(fp," %lf",cur->m_labelStats[i]);
          }
          cur->m_bestTest.writeTest(fp);
          fprintf(fp,"\n");
          que.pop();
          que.push( cur->m_leftChildNode );
          que.push( cur->m_rightChildNode );

          /* if( cur->m_leftChildNode != nullptr){
            que.push( cur->m_leftChildNode );
          }
          if( cur->m_rightChildNode != nullptr){
            que.push( cur->m_rightChildNode );
          } */
        }
      }
    }

    void loadNode(OnlineNode *node, FILE *fp)
    {
      char node_type;
      queue<OnlineNode *> tree;
      OnlineNode *cur;
      tree.push(node);

      while(!tree.empty()){
        cur = tree.front();
        tree.pop();

        fscanf(fp,"%c ",&node_type);

        if(node_type == 'T'){
          //cout<<"Loading next tree"<<endl;
          return;
        } else if (node_type == 'L'){
          fscanf(fp, "%d ", &cur->m_depth);
          fscanf(fp, "%d ", &cur->m_isLeaf);
          fscanf(fp, "%lf ", &cur->m_counter);
          fscanf(fp, "%d ", &cur->m_label);
          fscanf(fp, "%lf ", &cur->m_parentCounter);
          for (int i = 0; i < cur->m_labelStats.size(); i++) {
            fscanf(fp, "%lf ", &cur->m_labelStats[i]);
          }
          for (int i = 0; i < cur->m_onlineTests.size(); i++) {
            cur->m_onlineTests[i].loadTest(fp);
          }
          fscanf(fp,"\n");

        } else if (node_type == 'S'){
          fscanf(fp, "%d ", &cur->m_depth);
          fscanf(fp, "%d ", &cur->m_isLeaf);
          fscanf(fp, "%lf ", &cur->m_counter);
          fscanf(fp, "%d ", &cur->m_label);
          fscanf(fp, "%lf ", &cur->m_parentCounter);
          for (int i = 0; i < cur->m_labelStats.size(); i++) {
            fscanf(fp, "%lf ", &cur->m_labelStats[i]);
          }
          HyperplaneFeature temp_bestTest(*m_numClasses, *m_numFeatures, m_hp->numProjectionFeatures, *m_minFeatRange, *m_maxFeatRange);
          cur->m_bestTest = temp_bestTest;
          cur->m_bestTest.loadTest(fp);
          fscanf(fp,"\n");

          pair<vector<double> , vector<double> > parentStats = cur->m_bestTest.getStats();
          cur->m_leftChildNode = new OnlineNode(*cur->m_hp, *cur->m_numClasses, *cur->m_numFeatures, *cur->m_minFeatRange, *cur->m_maxFeatRange, cur->m_depth + 1,
                  parentStats.second);
          tree.push(cur->m_leftChildNode);
          cur->m_rightChildNode = new OnlineNode(*cur->m_hp, *cur->m_numClasses, *cur->m_numFeatures, *cur->m_minFeatRange, *cur->m_maxFeatRange, cur->m_depth + 1,
                  parentStats.first);
          tree.push(cur->m_rightChildNode);
        }
      }
      return;
    }

private:
    const int *m_numClasses;
    const int *m_numFeatures;
    int m_depth;
    bool m_isLeaf;
    double m_counter;
    int m_label;
    double m_parentCounter;
    const Hyperparameters *m_hp;
    const vector<double> *m_minFeatRange;
    const vector<double> *m_maxFeatRange;

    vector<double> m_labelStats;

    OnlineNode* m_leftChildNode;
    OnlineNode* m_rightChildNode;

    vector<HyperplaneFeature> m_onlineTests;
    HyperplaneFeature m_bestTest;

    bool shouldISplit() {
        bool isPure = false;
        for (int i = 0; i < *m_numClasses; i++) {
            if (m_labelStats[i] == m_counter + m_parentCounter) {
                isPure = true;
                break;
            }
        }

        if (isPure) {
            return false;
        }

        if (m_depth >= m_hp->maxDepth) { // do not split if max depth is reached
            return false;
        }

        if (m_counter < m_hp->counterThreshold) { // do not split if not enough samples seen
            return false;
        }

        return true;
    }

};

#endif /* ONLINENODE_H_ */
