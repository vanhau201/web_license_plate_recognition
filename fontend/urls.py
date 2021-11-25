from django.urls import path
from django.conf import settings

from django.conf import settings

from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('statistics/', views.statistics, name='statistics'),

]
