## Used-Car-Price-Predictor

<div id="header">
  <h1>
    Used-car-Price-Predictor<p></p><p></p><p></p>
    <img src="Screenshots\priceprediction_webpage.jpg" alt="price predictor screenshot" width="1000" align="center"/>
  </h1>
</div>

This project aims to predict price of a used car. It uses regression algorithms such as, Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBoost, CatBoost and AdaBoost to predict price. Best model is chosen after evaluation with a minimum threshold score of 60% and if none of the models cross the threshold score then a log of "no best model found" is created.

A prediction pipeline is created using flask web app. Further I deployed my machine learning application using GitHub actions in Azure web app. 

<div id="line">
  <h1><p></p>
    <img src="Screenshots\azuredeployment.jpg" alt="app deployment" width="1000" align="center"/>
  </h1>
</div>

## To use the price predictor application

1. Clone the repository using the command: `git clone <repo_url>`
2. Navigate to the project directory: `cd <repo>`
3. Run the script using the command: `python app.py`
4. Once the application is running.
    Use default port of 127.0.0.1.5000 - for home page
    127.0.0.1.5000/predictprice - for predictor page

