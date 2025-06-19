def export_model(model,export_dir="exported_model"):
    model.export_model(export_dir = export_dir)
    print(f"Model exported to {export_dir}")
