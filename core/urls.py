from django.urls import path
from .views import HomeTemplateView
from .views import DiseaseView
from . import views
urlpatterns = [
    path('', HomeTemplateView.as_view()),
    path('disease',DiseaseView.as_view()),
    path('result',views.result,name='result'),
]
