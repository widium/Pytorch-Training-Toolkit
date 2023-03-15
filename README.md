### Track And Analyze Training Process of Pytorch Model :
- [Display Tracked Value While Training Loop](#display-tracked-information-in-history-while-training)
- [Diagnostic Training Loop](#diagnostic-training-process)
- [Plot Training/Validation Curves](#plot-training--validation-curves)
***
![](https://i.imgur.com/9IYr2DY.png)
![](https://i.imgur.com/VRCesZo.png)
***
Use It in Pytorch Training Loop
- **Initialize Instance** Before Loop :
~~~python
from Tracker import HistoricalTraining

# create a HistoricalTraining instance
history = HistoricalTraining(max_epochs=NBR_EPOCHS)

# Initialize List for each tracked Value
history["Train BCE"] = list()
history["Validation BCE"] = list()
history["Train Accuracy"] = list()
history["Validation Accuracy"] = list()
~~~
**Use** Instance method in Loop :
- **`display_info()`** display with reccurance all Tracked Value
- **`diagnostic()`** Compute and Display Diagnostic of Model (OverFitting and UnderFitting), Give at method the Target metric to Diagnostic `metric_name`
- **`plot_curves`** Visualize Tracked Value in Pretty Subplot, Give at method the Target Loss and Metric to Plot `loss_name` and `metric_name`
~~~python
for epoch in range(NBR_EPOCHS):

    # Activate Training Mode
    model.train()
    
	# forward, convert logits...
	
    # Get Value
    train_loss = loss_function(y_logits, y_train)
    train_score = metric_function(y_preds, y_train)
    
    # Compute Gradients, Optimize, ect...
    
    # Activate Evaluation Mode
    model.eval()

    # Make Accelerate Prediction in Inference Mode
    with torch.inference_mode():

        # forward, convert logits...
        
        # Get Value
        val_loss = loss_function(y_logits, y_test)
        val_score = metric_function(y_preds, y_test)

    # Tracked all New Value in Instance
    # Convert all Value to Numpy (detach from gpu)
    history["Train BCE"].append(to_numpy(val_loss))
    history["Train Accuracy"].append(to_numpy(train_score))
    history["Validation BCE"].append(to_numpy(train_loss))
    history["Validation Accuracy"].append(to_numpy(val_score))

	# print information with current epoch and reccurance
    history.display_info(current_epoch=epoch,
                          reccurance=100)

# At the end of The loop Diagnostic and Visualize Training Process
history.diagnostic(average=False)
history.diagnostic(average=True)
history.plot_curves(loss_name="BCE")
~~~
***
## Display Tracked Information in `history` While Training

- **Initialize** Instance Before Loop :
~~~python
from Tracker import HistoricalTraining

# create a HistoricalTraining instance
history = HistoricalTraining(max_epochs=NBR_EPOCHS)

# Initialize List for each tracked Value
history["Train BCE"] = list()
history["Validation BCE"] = list()
history["Train Accuracy"] = list()
history["Validation Accuracy"] = list()
~~~
**Use** Instance method in Loop :
- **`display_info()`** display with reccurance all Tracked Value
	- give the `current_epoch`
	- give the `reccurance` of display
~~~python
for epoch in range(NBR_EPOCHS):

    # Activate Training Mode
    model.train()
    
	# forward, convert logits...
	
    # Get Value
    train_loss = loss_function(y_logits, y_train)
    train_score = metric_function(y_preds, y_train)
    
    # Compute Gradients, Optimize, ect...
    
    # Activate Evaluation Mode
    model.eval()

    # Make Accelerate Prediction in Inference Mode
    with torch.inference_mode():

        # forward, convert logits...
        
        # Get Value
        val_loss = loss_function(y_logits, y_test)
        val_score = metric_function(y_preds, y_test)

    # Tracked all New Value in Instance
    # Convert all Value to Numpy (detach from gpu)
    history["Train BCE"].append(to_numpy(val_loss))
    history["Train Accuracy"].append(to_numpy(train_score))
    history["Validation BCE"].append(to_numpy(train_loss))
    history["Validation Accuracy"].append(to_numpy(val_score))

	# print information with current epoch and reccurance
    history.display_info(current_epoch=epoch,
                          reccurance=100)
~~~
![](https://i.imgur.com/6AD6Tt9.png)
***
## Diagnostic Training Process
Use at the End or While Pytorch Training Loop
-  **Initialize Instance** Before Loop :
~~~python
from Tracker import HistoricalTraining

# create a HistoricalTraining instance
history = HistoricalTraining(max_epochs=NBR_EPOCHS)

# Initialize List for each tracked Value
history["Train BCE"] = list()
history["Validation BCE"] = list()
history["Train F1Score"] = list()
history["Validation F1Score"] = list()
~~~
**Use** Instance method in Loop :
- **`diagnostic()`** Compute and Display Diagnostic of Model (OverFitting and UnderFitting), 
- **Give** at method the **Target metric** to Diagnostic `metric_name`
~~~python
for epoch in range(NBR_EPOCHS):

    # Activate Training Mode
    model.train()
    
	# forward, convert logits...
	
    # Get Value
    train_loss = loss_function(y_logits, y_train)
    train_score = metric_function(y_preds, y_train)
    
    # Compute Gradients, Optimize, ect...
    
    # Activate Evaluation Mode
    model.eval()

    # Make Accelerate Prediction in Inference Mode
    with torch.inference_mode():

        # forward, convert logits...
        
        # Get Value
        val_loss = loss_function(y_logits, y_test)
        val_score = metric_function(y_preds, y_test)

    # Tracked all New Value in Instance
    # Convert all Value to Numpy (detach from gpu)
    history["Train BCE"].append(to_numpy(val_loss))
    history["Train F1Score"].append(to_numpy(train_score))
    history["Validation BCE"].append(to_numpy(train_loss))
    history["Validation F1Score"].append(to_numpy(val_score))

# At the end of The loop Diagnostic and Visualize Training Process
history.diagnostic(average=False, metric_name="F1")
history.diagnostic(average=True, metric_name="F1")
~~~
![](https://i.imgur.com/F8sKjpR.png)
***
## Plot Training & Validation Curves 
Use at the End of Pytorch Training Loop
- **Initialize Instance** Before Loop :
~~~python
from Tracker import HistoricalTraining

# create a HistoricalTraining instance
history = HistoricalTraining(max_epochs=NBR_EPOCHS)

# Initialize List for each tracked Value
history["Train BCE"] = list()
history["Validation BCE"] = list()
history["Train F1Score"] = list()
history["Validation F1Score"] = list()
~~~
**Use** Instance method in Loop :
- **`plot_curves`** Visualize Tracked Value in Pretty Subplot, 
- **Give** at method the Target Loss and Metric to Plot `loss_name` and `metric_name`
~~~python
for epoch in range(NBR_EPOCHS):

    # Activate Training Mode
    model.train()
    
	# forward, convert logits...
	
    # Get Value
    train_loss = loss_function(y_logits, y_train)
    train_score = metric_function(y_preds, y_train)
    
    # Compute Gradients, Optimize, ect...
    
    # Activate Evaluation Mode
    model.eval()

    # Make Accelerate Prediction in Inference Mode
    with torch.inference_mode():

        # forward, convert logits...
        
        # Get Value
        val_loss = loss_function(y_logits, y_test)
        val_score = metric_function(y_preds, y_test)

    # Tracked all New Value in Instance
    # Convert all Value to Numpy (detach from gpu)
    history["Train BCE"].append(to_numpy(val_loss))
    history["Train F1Score"].append(to_numpy(train_score))
    history["Validation BCE"].append(to_numpy(train_loss))
    history["Validation F1Score"].append(to_numpy(val_score))

# At the end of The loop Visualize Training/ Validation Process
history.plot_curves(loss_name="BCE", metric_name="F1")
~~~
![](https://i.imgur.com/6QkermH.png)