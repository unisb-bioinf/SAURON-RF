# SAURON-RF
Welcome to the GitHub page of SAURON-RF (SimultAneoUs Regression and classificatiON Random Forests), a tool for drug sensitivity prediction in cancer based on the idea to perform a joint classification and regression analysis.

For issues and questions, please contact Kerstin Lenhof (klenhof[at]bioinf.uni-sb.de). If you use SAURON-RF or the code in this repository, please cite out paper. 

# Usage
SAURON-RF can be executed as a python3 script in the console. It requires a single json-config file (see example_Json_config.txt in Example_Data folder) as input.

Used python libraries:
json
numpy
math
time
sys
collections
scipy
sklearn

Example call:
python3 cv_main.py example_Json_config.json
