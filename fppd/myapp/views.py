from django.shortcuts import render,redirect
from . import fake
import pandas as pd
# Create your views here.

def main_page(request):
    return render(request,'main_page.html')

def random_forest_classifier_function_view(request):
    rf_accuracy,report_html,rf_c_matrix=fake.pred_rfc()
    context={
        "rf_accuracy":rf_accuracy,
        "rf_c_matrix":rf_c_matrix,
        "report_html":report_html
    }
    return render(request,'random_forest_classifier.html',context)

def naive_bayes_classifier_function(request):
    nb_accuracy,nb_report_html,nb_c_matrix=fake.pred_nb()
    context={
        "nb_accuracy":nb_accuracy,
        "nb_report_html":nb_report_html,
        "nb_c_matrix":nb_c_matrix
    }
    return render(request,'naive_bayes_classifier.html',context)

def linear_regression(request):
    mse,c_matrix=fake.pred_linear_regression()
    context={
        "mse":mse,
        "c_matrix":c_matrix
    }
    return render(request,'linear_regression.html',context)

def logistic_regression_function(request):
    nb_accuracy,nb_report_html,nb_c_matrix=fake.pred_nb()
    context={
        "nb_accuracy":nb_accuracy,
        "nb_report_html":nb_report_html,
        "nb_c_matrix":nb_c_matrix
    }
    return render(request,'logistic_regression.html',context)

def lstm_dataset_pred(request):
    test_loss,test_accuracy,selected_features,result,c_matrix,report_html=fake.lstm_approach()
    
    context={
        "test_loss":test_loss*100,
        "test_accuracy":test_accuracy*100,
        "selected_features":selected_features,
        "result":result,
        "c_matrix":c_matrix,
        "report_html":report_html,  
    }
    return render(request,'lstm_prediction.html',context)

# def lstm_id(request):
#     from .id_form import id_form
#     if request.method == 'POST':
#         form=id_form(request.POST)
#         if form.is_valid():
#             form_id_input=form.cleaned_data['aid']
#             form_id=int(form_id_input)
#             features,target,specific_features,loss,accuracy,result=fake.lstm_id(form_id)

#     context={
#         "form":form,
#         "features":features,
#         "target":target,
#         "specific_features":specific_features,
#         "loss":loss,
#         "accuracy":accuracy,
#         "result":result
#     }
#     return render(request,"lstm_id.html",context)

def lstm_id(request):
    from .id_form import id_form
    
    # Initialize variables
    # features = None
    # target = None
    # specific_features = None
    # loss = None
    # accuracy = None
    result = None
    form = id_form()  # Initialize the form to avoid 'referenced before assignment'

    if request.method == 'POST':
        form = id_form(request.POST)
        if form.is_valid():
            form_id_input = form.cleaned_data['aid']
            form_id = int(form_id_input)

            # Call the lstm_id function (assuming it's imported as 'fake.lstm_id')
            result = fake.lstm_id(form_id)

    context = {
        "form": form,
        "result": result
    }
    
    return render(request, "lstm_id.html", context)
