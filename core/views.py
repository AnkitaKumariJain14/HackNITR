from django.shortcuts import render,redirect
from django.views.generic import TemplateView
from .models import *
from django.http import HttpResponse

class HomeTemplateView(TemplateView):
    template_name = 'home.html'

    # override get context date method
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)  # first, call super get context data
        context['about'] = About.objects.first()
        context['services'] = Service.objects.all()
        context['works'] = RecentWork.objects.all()
        return context

class DiseaseView(TemplateView):
    template_name = 'disease.html'


def result(request):
    return render(request,"result.html")