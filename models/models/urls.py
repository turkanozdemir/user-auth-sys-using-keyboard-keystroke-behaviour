"""
URL configuration for models project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path


from django.urls import path
from models_app.views import  svm_predictions, mlp_predictions, knn_predictions, lr_predictions, all_models


urlpatterns = [
    path('svm-predictions/', svm_predictions, name='svm-predictions'),
    path('mlp-predictions/', mlp_predictions, name='mlp-predictions'),
    path('knn-predictions/', knn_predictions, name='knn-predictions'),
    path('lr-predictions/', lr_predictions, name='lr-predictions'),
    path('all-models/', all_models, name='all-models')
]
