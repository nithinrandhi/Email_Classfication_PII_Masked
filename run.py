from models import load_model, predict_category

model = load_model()

sample_email = "Hi, I need help with my last invoice. The amount seems incorrect."
predicted_class = predict_category(sample_email, model)

print("Predicted Category:", predicted_class)
