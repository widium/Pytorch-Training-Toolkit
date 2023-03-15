# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Tracker.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/03/15 16:20:34 by ebennace          #+#    #+#              #
#    Updated: 2023/03/15 16:20:56 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

class HistoricalTraining(dict):
    """Store, Diagnostic and Visualize all Data in Training Loop

    Args:
        dict : Heritage of Python Dictionary
    """
    def __init__(self,
                max_epochs : int):
        """Initialize Class with Maximum Epochs in Training Loop
           This will create a List of epochs for visualization

        Args:
            max_epochs (int): max_epochs
        """
        super().__init__()

        self.max_epochs = max_epochs
        self["Epochs"] = list()

    # ========================================================================================== #
        
    def display_info(self, 
                    current_epoch : int,
                    reccurance : int=1):
        """Display Information Sequentialy for each Epochs Like a Table
           Place this function in loop and set the reccurance for control print
           the width of print scale dynamically with name and nbr of Value to Track

        Args:
            current_epoch (int): current_epoch
            reccurance (int, optional): Reccurance of display information. Defaults to 1.
        """

        self["Epochs"].append(current_epoch)

        # ******************** Create Header ********************#
        
        # Print the name of tracked value as Header
        if current_epoch == 0:

            headers = [f"{header:<18}|" for header in self.keys()]
            self.nbrChar = len(''.join(headers))

            print(*headers)
            print(f"{'=' * self.nbrChar}")
        
        # ******************** Check Recurrance ********************#
        # Print only reccurent epoch
        if (current_epoch % reccurance != 0):
            # Let pass the last epoch
            if (current_epoch + 1 == self.max_epochs):
                pass
            else:
                return
        
        # ******************** Get Tracked Value ********************#
        
        # Get the Last Value of Tracked Value and print them proprely
        lastValue = list()

        # loop over each tracked value in Class Dictionary
        for key in self:
            
            # Print Epochs Value with Progress Rate
            if key == "Epochs":
                percentage = ((current_epoch)* 100) / self.max_epochs
                value = f"{percentage}% [{current_epoch + 1}/{self.max_epochs}]"
                value = f"{value:<18}|"

            # Get the Last Value of each Tracked Value
            else:
                value = f"{self[key][-1]:.4f}"
                value = f"{value:<18}|"
            
            lastValue.append(f"{value}")
        
        # ******************** Display Tracked Value ********************#
        
        # Print Last Tracked Value and Separate them with new Line scaled with Header   
        print(*lastValue)
        print(f"{'-' * self.nbrChar}")

    # ========================================================================================== #
    
    def diagnostic(self,
                    metric_name="Accuracy",
                    average=False,
                    returning=False):
        """Create and Visualize Diagnostic of Training Loop Tracked Value
           Computing Programatically OverFitting and UnderFitting (Bias & Variance)
           of Model Performance.
           Analyze the Target metric Name passed in argument

        Args:
            metric_name (str, optional): Target metric to analyze. Defaults to "Accuracy".
            average (bool, optional): Compute the Average of Metric Value in Entire Training Process. Defaults to False.
            returning (bool, optional): Return a dictionary of diagnostic. Defaults to False.

        Returns:
            dict: dictionary of diagnostic
        """
    
        # ******************** Setup Constant ********************#
        
        # Setup Constant Different for Average Diagnostic
        if average == True:
            BIAS = 0.4 # below this value is High Bias
            LOW_BIAS = 0.65 # High Accuracy / Low Bias

        else :
            BIAS = 0.80 # below this value is High Bias
            LOW_BIAS = 0.95 # High Accuracy / Low Bias

        VARIANCE = 0.10 # above this value is High Variance
        LOW_VARIANCE = 0.05 # below this value is Low Variance

        # ******************** Recover Value ********************#
        
        # remove maj for compare name
        metric_name = metric_name.lower()

        # Get the key of target metric name in Train and Validation Data
        for key in self:

            if ("val" in key.lower()) and (metric_name in key.lower()):

                    val_score_name = key
                    
                    # Compute the mean of get the last value
                    if average == True:
                        validation_score = np.mean(self[key])
                    else :
                        validation_score = self[key][-1]
                
            elif ("train" in key.lower()) and (metric_name in key.lower()):

                    train_score_name = key
                    
                    # Compute the mean of get the last value
                    if average == True:
                        train_score = np.mean(self[key])
                    else :
                        train_score = self[key][-1]

        # ******************** Diagnostic Value ********************#

        # Storage
        status = dict()
        bias = list()
        variance = list()

        # Compute the Bias
        if (validation_score < BIAS):
            bias.append("High Bias")
            bias.append("UnderFitting")
        
        elif (validation_score > BIAS) and (validation_score < LOW_BIAS):
            bias.append("Medium Bias")

        elif (validation_score >= LOW_BIAS):
            bias.append("Low Bias")  
            bias.append("Good Accuracy !")  

        # Compute The Variance
        gap = train_score - validation_score

        if (gap > VARIANCE):
            variance.append("High Variance")
            variance.append("Overfitting")
        
        elif (gap < VARIANCE and gap > LOW_VARIANCE):
            variance.append("Medium Variance")
                
        elif (gap <= LOW_VARIANCE):
            variance.append("Low Variance")
            variance.append("Good Generalization !")  
        
        # Store Diagnostic in Dictionary
        status["Bias and UnderFitting"] = bias
        status["Variance and OverFitting"] = variance
        status[train_score_name] = f"{train_score:.3f}"
        status[val_score_name] = f"{validation_score:.3f}"
        
        # Return Diagnostic Dictionary or Display Information
        if (returning == True):
            return (status)
        
        # ******************** Display Information ********************#
        print(f"===== {'Average' if average == True else 'Last'} Diagnostic =====")

        print(f"\nBias and UnderFitting :\n")
        for s in status["Bias and UnderFitting"]:
            print(f"- {s}") 

        print(f"\nVariance and OverFitting :\n")
        for s in status["Variance and OverFitting"]:
            print(f"- {s}") 

        print(f"\nValue :\n")
        if average == True:
            print(f"Average {train_score_name} : {status[train_score_name]}")
            print(f"Average {val_score_name} : {status[val_score_name]}")
        else :
            print(f"Last {train_score_name} : {status[train_score_name]}")
            print(f"Last {val_score_name} : {status[val_score_name]}")
        
        print("=" * 40)
        print("\n")

    # ========================================================================================== #
    
    def plot_curves(self,
                    metric_name="Accuracy",
                    loss_name="Loss"):
        """Visualize Loss and Metric Curve 
          for Target loss and Metric Tracked Value
           
        Args:
            metric_name (str, optional): Target Metric Name. Defaults to "Accuracy".
            loss_name (str, optional): Target Loss Name. Defaults to "Loss".
        """
        
        # ******************** Get Target Tracked Value ********************#
        
        # remove maj for compare name
        metric_name = metric_name.lower()
        loss_name = loss_name.lower()
        
        # loop over key's tracked value
        for key in self:

            # Get Key with Loss Name for Training and Validation #
            if ("val" in key.lower()) and (loss_name in key.lower()):
                    
                    validation_loss_name = key
                    validation_loss = self[key]
                
            elif ("train" in key.lower()) and (loss_name in key.lower()):

                    train_loss_name = key
                    train_loss = self[key]

            # Get Key with Metric Name for Training and Validation #
            if ("val" in key.lower()) and (metric_name in key.lower()):
                    
                    validation_score_name = key
                    validation_score = self[key]
                
            elif ("train" in key.lower()) and (metric_name in key.lower()):

                    train_score_name = key
                    train_score = self[key]
        
        # ******************** Create Subplot ********************#
        
        plt.figure(figsize=(20, 12))

        plt.subplot(1, 2, 1)
        plt.plot(self["Epochs"], train_loss, label=f"{train_loss_name}")
        plt.plot(self["Epochs"], validation_loss, label=f"{validation_loss_name}")
        plt.title(f"{loss_name.capitalize()}", fontsize=15)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel(f"Value of {loss_name.capitalize()}", fontsize=15)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self["Epochs"], train_score, label=f"{train_score_name}")
        plt.plot(self["Epochs"], validation_score, label=f"{validation_score_name}")
        plt.title(f"{metric_name.capitalize()}", fontsize=15)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel(f"Value of {metric_name.capitalize()}", fontsize=15)
        plt.legend()
    # ========================================================================================== #