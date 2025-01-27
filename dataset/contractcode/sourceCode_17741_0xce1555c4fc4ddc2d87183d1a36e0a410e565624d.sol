/**
 *Submitted for verification at Etherscan.io on 2016-06-19
*/

contract storadge {
    event log(string description);
	function save(
        string mdhash
    )
    {
        log(mdhash);
    }
}