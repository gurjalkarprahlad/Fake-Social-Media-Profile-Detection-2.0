from django import forms

class id_form(forms.Form):
    aid=forms.CharField(label="Account ID")