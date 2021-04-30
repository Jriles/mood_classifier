from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train_model_button', views.train_model, name='train_model_button'),
]
