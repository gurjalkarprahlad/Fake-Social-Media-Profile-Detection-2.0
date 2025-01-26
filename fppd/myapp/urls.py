from django.urls import path,include
from . import views

urlpatterns=[
    path('',views.main_page),
    path('random_forest_classifier/',views.random_forest_classifier_function_view,name='random_forest_classifier'),
    path('naive_bayes_classifier/',views.naive_bayes_classifier_function,name='naive_bayes_classifier'),
    path('linear_regression/',views.linear_regression,name="linear_regression"),
    path('logistic_regression/',views.logistic_regression_function,name="logistic_regression"),
    path('lstm/',views.lstm_dataset_pred,name='lstm'),
    path('lstm_id/',views.lstm_id,name="lstm_id")

]