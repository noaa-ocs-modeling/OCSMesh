import unittest
from unittest.mock import patch
import sys

class TestMPIExecutor(unittest.TestCase):
    
    def test_is_manager_without_mpi(self):
        """Test that is_manager() safely returns True when mpi4py is not installed."""
        
        import ocsmesh.mpi
        from ocsmesh.mpi import MPIExecutor
        
        # Save the original state of the global variables
        original_flag = ocsmesh.mpi._MPI_IMPORT_ATTEMPTED
        original_mpi = ocsmesh.mpi._MPI
        
        try:
            # Simulate an environment where mpi4py is not installed
            with patch.dict('sys.modules', {'mpi4py': None}):
                # Reset the internal flags so _get_mpi() tries to re-import
                ocsmesh.mpi._MPI_IMPORT_ATTEMPTED = False
                ocsmesh.mpi._MPI = None
                
                self.assertTrue(MPIExecutor.is_manager())
                
        finally:
            # Safely restore the original state so we don't break other tests
            ocsmesh.mpi._MPI_IMPORT_ATTEMPTED = original_flag
            ocsmesh.mpi._MPI = original_mpi




if __name__ == '__main__':
    unittest.main()