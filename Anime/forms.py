from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import MyUser

class UserCreationForm(UserCreationForm):
    class Meta:
        model = MyUser
        fields = ['username', 'password1', 'password2']