from django.urls import path
from . import views


urlpatterns = [
    path('home/',views.view1,name="view1"),
]