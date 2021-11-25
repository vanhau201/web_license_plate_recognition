from django.shortcuts import render

# Create your views here.


def home(request):
    return render(request, "home.html", {
        'nav': 'home'
    })


def statistics(request):
    return render(request, "statistics.html", {
        'nav': 'statistics'
    })
