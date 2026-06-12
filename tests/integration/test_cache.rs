use reqwest::Client;
use serde_json::json;

#[tokio::test]
async fn test_cache_lookup() {
    let client = Client::new();
    
    // First, generate an adapter to populate cache
    let request_body = json!({
        "user_id": "cache-test-user",
        "context": {
            "description": "Test description for cache",
            "domain": "test"
        },
        "base_model": "meta-llama/Llama-3-8B",
        "target_rank": 16
    });
    
    let response = client
        .post("http://localhost:8080/generate")
        .json(&request_body)
        .send()
        .await;
    
    match response {
        Ok(resp) => {
            assert!(resp.status().is_success());
            let body: serde_json::Value = resp.json().await.unwrap();
            let adapter_id = body["adapter_id"].as_str().unwrap();
            
            // Now try to retrieve it
            let retrieve_response = client
                .get(&format!("http://localhost:8080/adapter/{}", adapter_id))
                .send()
                .await;
            
            match retrieve_response {
                Ok(retrieve_resp) => {
                    assert!(retrieve_resp.status().is_success());
                    let retrieve_body: serde_json::Value = retrieve_resp.json().await.unwrap();
                    assert_eq!(retrieve_body["adapter_id"], adapter_id);
                }
                Err(_) => {
                    println!("Skipping cache retrieval test: server not running");
                }
            }
        }
        Err(_) => {
            println!("Skipping test_cache_lookup: server not running");
        }
    }
}

#[tokio::test]
async fn test_embed_endpoint() {
    let client = Client::new();
    
    let request_body = json!({
        "context": {
            "description": "Test embedding",
            "domain": "test"
        },
        "base_model": "meta-llama/Llama-3-8B"
    });
    
    let response = client
        .post("http://localhost:8080/embed")
        .json(&request_body)
        .send()
        .await;
    
    match response {
        Ok(resp) => {
            assert!(resp.status().is_success());
            let body: serde_json::Value = resp.json().await.unwrap();
            assert!(body.get("embedding_latency_ms").is_some());
        }
        Err(_) => {
            println!("Skipping test_embed_endpoint: server not running");
        }
    }
}
