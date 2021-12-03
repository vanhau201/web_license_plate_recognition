from django.shortcuts import render, redirect
from django.contrib.auth import (
    authenticate, update_session_auth_hash, get_user_model)
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
# Create your views here.


def login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        account = authenticate(username=username, password=password)
        if account != None:
            auth_login(request, account)
            return redirect("home")
        else:
            messages.warning(request, "Incorrect account or password")

    return render(request, "login.html")


def register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if username == '' or email == '' or password1 == '' or password2 == '':
            messages.warning(request, "Can not be empty")
        else:
            user = User.objects.filter(username=username)
            if len(user) > 0:
                messages.warning(request, "Username already exists")
            else:
                if len(password1) < 8:
                    messages.warning(
                        request, "The password is too easy to guess")
                else:
                    if password1 != password2:
                        messages.warning(
                            request, "Password is not correct")
                    else:
                        userNew = User.objects.create_user(username=username, password=password1,
                                                           email=email, is_staff=False)
                        userNew.save()
                        messages.success(
                            request, "Registered successfully")

    return render(request, "register.html")


def logout(request):
    auth_logout(request)
    return redirect("home")


@login_required
def home(request):
    acc = authenticate(username="admi", password="12456")
    print(acc)
    return render(request, "home.html", {
        'nav': 'home'
    })


@login_required
def statistics(request):
    return render(request, "statistics.html", {
        'nav': 'statistics'
    })
