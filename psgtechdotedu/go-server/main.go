package main

import (
	"fmt"
	"net/http"
)

func main() {
  r := http.NewServeMux()

  r.HandleFunc("GET /", home)
  r.HandleFunc("GET /ping", health)
  r.HandleFunc("GET /health", health)

  if err := http.ListenAndServe(":8000", r); err == nil {
    fmt.Println("blud, there was some error in trying to server the damn router.")
  }
}

func home(w http.ResponseWriter, r *http.Request) {
  fmt.Fprintln(w, "sup, you've reached the psgtechdotedu new backend.")
}

func health(w http.ResponseWriter, r *http.Request) {
  fmt.Fprintln(w, "we healthy! or should i say pong!")
}

