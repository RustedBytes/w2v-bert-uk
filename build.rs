fn main() {
    println!("cargo:rerun-if-changed=src/csharp.rs");

    if std::env::var_os("CARGO_FEATURE_CSHARP").is_some() {
        csbindgen::Builder::default()
            .input_extern_file("src/csharp.rs")
            .csharp_dll_name("w2v_bert_uk")
            .csharp_namespace("W2vBertUk.Native")
            .csharp_class_name("NativeMethods")
            .csharp_class_accessibility("public")
            .generate_csharp_file("csharp/NativeMethods.g.cs")
            .expect("failed to generate C# bindings");
    }

    if std::env::var_os("CARGO_FEATURE_NODEJS").is_some() {
        napi_build::setup();
    }

    if std::env::var_os("CARGO_FEATURE_PYTHON").is_some() {
        pyo3_build_config::add_extension_module_link_args();
    }
}
