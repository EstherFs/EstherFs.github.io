from week10project import views
from django.urls import path

urlpatterns = [
    path("", views.home),
    path("ccu410410022", views.ccu410410022_function)
]
