# forms.py

from django import forms

class KeywordSelectionForm(forms.Form):
    keywords = forms.MultipleChoiceField(
        choices=[],  # This will be populated in the view
        widget=forms.CheckboxSelectMultiple,
        required=True
    )

    def clean_keywords(self):
        selected_keywords = self.cleaned_data['keywords']
        if len(selected_keywords) != 5:
            raise forms.ValidationError("You must select exactly 5 keywords.")
        return selected_keywords
