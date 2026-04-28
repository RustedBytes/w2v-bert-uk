package rustasr

import "testing"

func TestBoolOptionZeroValueUsesNativeDefault(t *testing.T) {
	var option BoolOption
	if option != BoolDefault {
		t.Fatalf("zero BoolOption = %d, want BoolDefault", option)
	}
}
