

from django.urls import path
from django.conf import settings

from django.conf import settings

from . import views
urlpatterns = [
    path('', views.apiOverview),
    path('list_image/', views.listImage),
    path('list_image_to_day/', views.listImageToDay),
    path('checkin/', views.checkIn),
    path('checkout/', views.checkOut),
    path('detail/<str:pk>/', views.detailImage),
    path('delete/<str:pk>/', views.deleteImage),
    path('update_check_in/<str:pk>/', views.updateCheckIn),
    path('update_check_out/', views.updateCheckOut),
    path('repair/<str:pk>/', views.repairImage),
    path('statistics/', views.statistics),

]
